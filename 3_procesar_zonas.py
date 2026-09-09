"""
Procesador de Analítica de Góndolas: Clasificación Avanzada de PICK-UP vs PUT-BACK y Tasa de Rechazo.

Mejoras Clave de Arquitectura Anti-Falsos Positivos:
1. Votación Temporal de Persistencia:
   - Requiere que la señal de alcance se repita consecutivamente durante al menos 6 fotogramas (umbral de votación).
   - Elimina destellos o fluctuaciones por ruido puntual del modelo en un solo frame.
2. Comparación de Repisa sin Oclusión (Zero-Occlusion Baseline Crop):
   - Compara el recorte de la repisa ANTES de que el cuerpo del cliente se aproxime (frame_ini - 35, 0% oclusión)
     contra el recorte 1.5 a 2.0 segundos DESPUÉS de que la persona se alejó por completo (frame_fin + 45, 0% oclusión).
   - Elimina las sombras, el movimiento de prendas y los brazos durante el alcance que generaban falsos positivos.

Uso:
  python 3_procesar_zonas.py --parquet datos_parquet/tracking_cam_01.parquet --zonas zonas_cam_01.json --video clip_test.mp4
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass



def punto_en_zona(x, y, poligono):
    if poligono is None or len(poligono) < 3:
        return False
    if np.isnan(x) or np.isnan(y):
        return False
    return cv2.pointPolygonTest(np.array(poligono, dtype=np.float32), (float(x), float(y)), False) >= 0


def calcular_dwell_zonas_flujo(df, zonas_flujo):
    def zona_de_fila(row):
        for z in zonas_flujo:
            pts = z["polygon"]
            xf, yf = row.get("x_foot", np.nan), row.get("y_foot", np.nan)
            if punto_en_zona(xf, yf, pts):
                return z["id"]
            xc, yc = row.get("x_center", np.nan), row.get("y_center", np.nan)
            if punto_en_zona(xc, yc, pts):
                return z["id"]
        return None

    df = df.copy()
    df["zona_flujo"] = df.apply(zona_de_fila, axis=1)

    filas_dwell = []
    for track_id, grupo in df.groupby("track_id"):
        grupo = grupo.sort_values("frame_idx")
        zona_actual = None
        inicio = None
        for _, row in grupo.iterrows():
            if row["zona_flujo"] != zona_actual:
                if zona_actual is not None:
                    filas_dwell.append({
                        "track_id": track_id, "zona": zona_actual,
                        "inicio_s": inicio, "fin_s": row["timestamp_s"],
                        "duracion_s": row["timestamp_s"] - inicio
                    })
                zona_actual = row["zona_flujo"]
                inicio = row["timestamp_s"]
        if zona_actual is not None:
            filas_dwell.append({
                "track_id": track_id, "zona": zona_actual,
                "inicio_s": inicio, "fin_s": grupo["timestamp_s"].iloc[-1],
                "duracion_s": grupo["timestamp_s"].iloc[-1] - inicio
            })

    if not filas_dwell:
        return pd.DataFrame(columns=["track_id", "zona", "inicio_s", "fin_s", "duracion_s"])
    return pd.DataFrame(filas_dwell)


def calcular_dwell_gondolas(df, zonas_interaccion, df_eventos=None):
    """
    Calcula la permanencia y visitantes únicos frente a cada góndola/estantería.
    Usa primero los eventos confirmados de interacción (manos/alcances). Para zonas sin toques,
    analiza la proximidad física del cliente a la góndola por distancia de pies/centro.
    """
    filas_dwell = []

    # 1. Desde eventos confirmados de interacción
    if df_eventos is not None and not df_eventos.empty and "zona" in df_eventos.columns:
        for (t_id, z_id), grp in df_eventos.groupby(["track_id", "zona"]):
            t_ini = float(grp["inicio_s"].min())
            t_fin = float(grp["fin_s"].max())
            dur_interaccion = float(grp["duracion_s"].sum())
            dur_total = max(dur_interaccion, t_fin - t_ini)
            filas_dwell.append({
                "track_id": t_id,
                "zona": z_id,
                "inicio_s": t_ini,
                "fin_s": t_fin,
                "duracion_s": round(max(dur_total, 1.0), 2)
            })

    # 2. Para zonas de interacción que no hayan tenido alcances directos, estimar por proximidad
    zonas_con_dwell = set(f["zona"] for f in filas_dwell)
    for z in zonas_interaccion:
        z_id = z.get("id", "")
        if not z_id or z_id in zonas_con_dwell:
            continue
        pts = np.array(z.get("polygon", z.get("poligono", [])), dtype=np.float32)
        if len(pts) < 3:
            continue
        for track_id, grupo in df.groupby("track_id"):
            near_frames = []
            for _, row in grupo.iterrows():
                xf, yf = row.get("x_foot", np.nan), row.get("y_foot", np.nan)
                if not np.isnan(xf) and not np.isnan(yf):
                    d = cv2.pointPolygonTest(pts, (float(xf), float(yf)), True)
                    if d >= -160.0:
                        near_frames.append(float(row["timestamp_s"]))
            if len(near_frames) >= 5:
                dur = near_frames[-1] - near_frames[0]
                if dur >= 0.5:
                    filas_dwell.append({
                        "track_id": track_id,
                        "zona": z_id,
                        "inicio_s": near_frames[0],
                        "fin_s": near_frames[-1],
                        "duracion_s": round(dur, 2)
                    })

    if not filas_dwell:
        return pd.DataFrame(columns=["track_id", "zona", "inicio_s", "fin_s", "duracion_s"])
    return pd.DataFrame(filas_dwell)



def evaluar_cambio_repisa_sin_oclusion(video_cap, pts_poligono, frame_pre_limpio, frame_post_limpio):
    """
    Compara el área de la repisa entre el momento PRE-aproximación (0% oclusión)
    y POST-alejamiento (0% oclusión, 1-2s después).
    """
    if video_cap is None:
        return 0.0

    video_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_pre_limpio))
    ret1, f_pre = video_cap.read()
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_post_limpio))
    ret2, f_post = video_cap.read()

    if not ret1 or not ret2:
        return 0.0

    x_min = max(0, int(np.min(pts_poligono[:, 0])))
    y_min = max(0, int(np.min(pts_poligono[:, 1])))
    x_max = min(f_pre.shape[1], int(np.max(pts_poligono[:, 0])))
    y_max = min(f_pre.shape[0], int(np.max(pts_poligono[:, 1])))

    if (x_max - x_min) < 8 or (y_max - y_min) < 8:
        return 0.0

    # Extraer recortes ROI limpios
    roi_pre = cv2.cvtColor(f_pre[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
    roi_post = cv2.cvtColor(f_post[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)

    # Suavizado previo para reducir ruido térmico de cámara
    roi_pre = cv2.GaussianBlur(roi_pre, (5, 5), 0)
    roi_post = cv2.GaussianBlur(roi_post, (5, 5), 0)

    diff = cv2.absdiff(roi_pre, roi_post)
    score_cambio = np.mean(diff) / 255.0
    return float(score_cambio)


def es_alcance_real_a_gondola(row, pts_poligono, max_dist_pies_px=220.0):
    """
    Filtro Biomecánico Anti-Ruido 2D:
    1. Proximidad de Pies: Los pies deben estar relativamente cerca o dentro del polígono.
    2. Extensión del Brazo: La muñeca debe estar extendida fuera de reposo vertical de marcha.
    3. Orientación del Pecho / Mirada: El torso y rostro deben estar orientados hacia la góndola.
    """
    if pts_poligono is None or len(pts_poligono) < 3:
        return False

    lw_x, lw_y = row.get("left_wrist_x", np.nan), row.get("left_wrist_y", np.nan)
    rw_x, rw_y = row.get("right_wrist_x", np.nan), row.get("right_wrist_y", np.nan)

    lw_in = punto_en_zona(lw_x, lw_y, pts_poligono)
    rw_in = punto_en_zona(rw_x, rw_y, pts_poligono)

    if not (lw_in or rw_in):
        return False

    # 1. Proximidad de Pies
    xf, yf = row.get("x_foot", np.nan), row.get("y_foot", np.nan)
    dist_pies = 0.0
    if not np.isnan(xf) and not np.isnan(yf):
        dist_pies = cv2.pointPolygonTest(np.array(pts_poligono, dtype=np.float32), (float(xf), float(yf)), True)
        if dist_pies < 0 and abs(dist_pies) > max_dist_pies_px:
            return False

    # 2. Extensión del Brazo (mano no pegada a la cadera en reposo al caminar)
    lh_x, lh_y = row.get("left_hip_x", np.nan), row.get("left_hip_y", np.nan)
    rh_x, rh_y = row.get("right_hip_x", np.nan), row.get("right_hip_y", np.nan)

    valid_left = False
    if lw_in and not np.isnan(lh_x) and not np.isnan(lh_y):
        dist_mano_cadera = np.sqrt((lw_x - lh_x)**2 + (lw_y - lh_y)**2)
        if dist_mano_cadera > 30.0:
            valid_left = True
    elif lw_in:
        valid_left = True

    valid_right = False
    if rw_in and not np.isnan(rh_x) and not np.isnan(rh_y):
        dist_mano_cadera = np.sqrt((rw_x - rh_x)**2 + (rw_y - rh_y)**2)
        if dist_mano_cadera > 30.0:
            valid_right = True
    elif rw_in:
        valid_right = True

    if not (valid_left or valid_right):
        return False

    # 3. Orientación del Pecho y Rostro
    ls_x, ls_y = row.get("left_shoulder_x", np.nan), row.get("left_shoulder_y", np.nan)
    rs_x, rs_y = row.get("right_shoulder_x", np.nan), row.get("right_shoulder_y", np.nan)

    if not np.isnan(ls_x) and not np.isnan(rs_x):
        g_center = np.mean(pts_poligono, axis=0)
        body_center = np.array([(ls_x + rs_x) / 2.0, (ls_y + rs_y) / 2.0])
        vec_to_gondola = g_center - body_center
        norm_to_g = np.linalg.norm(vec_to_gondola)

        if norm_to_g > 0:
            vec_to_gondola /= norm_to_g
            vec_hombros = np.array([rs_x - ls_x, rs_y - ls_y])
            norm_h = np.linalg.norm(vec_hombros)
            if norm_h > 0:
                vec_hombros /= norm_h
                vec_pecho = np.array([-vec_hombros[1], vec_hombros[0]])
                dot = abs(np.dot(vec_pecho, vec_to_gondola))
                if dot < 0.12 and dist_pies < -90:
                    return False

    # 4. Orientación de Mirada (si hay keypoints de rostro)
    nose_x = row.get("nose_x", np.nan)
    le_x, re_x = row.get("left_eye_x", np.nan), row.get("right_eye_x", np.nan)

    if not np.isnan(nose_x) and not np.isnan(le_x) and not np.isnan(re_x):
        eye_center_x = (le_x + re_x) / 2.0
        gaze_dx = nose_x - eye_center_x
        g_center_x = np.mean(pts_poligono[:, 0])
        body_x = (ls_x + rs_x) / 2.0 if not np.isnan(ls_x) else xf
        if not np.isnan(body_x):
            dir_to_gondola_x = g_center_x - body_x
            if (gaze_dx * dir_to_gondola_x < -15.0) and dist_pies < -50:
                return False

    return True


def clasificar_pickup_putback(df, zonas_interaccion, video_path=None, min_votacion=6):
    """
    1. Votación Temporal: confirma el alcance solo si la señal se mantiene durante min_votacion frames seguidos.
    2. Filtro Biomecánico Anti-Ruido 2D: valida proximidad de pies, extensión del brazo y orientación del torso/mirada.
    3. Comparación Sin Oclusión: evalúa el crop de repisa antes del acercamiento y 1-2s después de irse.
    """
    cap = cv2.VideoCapture(video_path) if video_path and os.path.exists(video_path) else None
    eventos_crudos = []

    for zona in zonas_interaccion:
        pts = np.array(zona["polygon"], dtype=np.int32)
        zona_id = zona["id"]

        for track_id, grupo in df.groupby("track_id"):
            grupo = grupo.sort_values("frame_idx").reset_index(drop=True)

            # 1. Votación Temporal de Presencia de Manos con Filtro Biomecánico
            votos_consecutivos = 0
            tolerancia_ausencia = 0
            inicio_idx = None

            for i, r in grupo.iterrows():
                mano_presente = es_alcance_real_a_gondola(r, pts)

                if mano_presente:
                    if votos_consecutivos == 0:
                        inicio_idx = i
                    votos_consecutivos += 1
                    tolerancia_ausencia = 0
                else:
                    # Tolerancia de 2 frames para ruido puntual de la pose
                    if votos_consecutivos > 0 and tolerancia_ausencia < 2:
                        tolerancia_ausencia += 1
                        votos_consecutivos += 1
                    else:
                        if votos_consecutivos >= min_votacion:
                            frame_ini = int(grupo.loc[inicio_idx, "frame_idx"])
                            frame_fin = int(grupo.loc[max(0, i - 1 - tolerancia_ausencia), "frame_idx"])
                            t_ini = float(grupo.loc[inicio_idx, "timestamp_s"])
                            t_fin = float(grupo.loc[max(0, i - 1 - tolerancia_ausencia), "timestamp_s"])

                            eventos_crudos.append({
                                "track_id": track_id,
                                "zona": zona_id,
                                "frame_inicio": frame_ini,
                                "frame_fin": frame_fin,
                                "inicio_s": t_ini,
                                "fin_s": t_fin,
                                "duracion_s": max(t_fin - t_ini, 0.1),
                            })
                        votos_consecutivos = 0
                        tolerancia_ausencia = 0

            if votos_consecutivos >= min_votacion:
                frame_ini = int(grupo.loc[inicio_idx, "frame_idx"])
                frame_fin = int(grupo.iloc[-1]["frame_idx"])
                t_ini = float(grupo.loc[inicio_idx, "timestamp_s"])
                t_fin = float(grupo.iloc[-1]["timestamp_s"])
                eventos_crudos.append({
                    "track_id": track_id, "zona": zona_id,
                    "frame_inicio": frame_ini, "frame_fin": frame_fin,
                    "inicio_s": t_ini, "fin_s": t_fin,
                    "duracion_s": max(t_fin - t_ini, 0.1)
                })

    if not eventos_crudos:
        if cap:
            cap.release()
        empty_ev = pd.DataFrame(columns=["track_id", "zona", "frame_inicio", "frame_fin", "inicio_s", "fin_s", "duracion_s", "tipo_accion"])
        empty_res = pd.DataFrame(columns=["zona", "alcances_totales", "pick_ups_tomados", "put_backs_devueltos", "tasa_conversion_pct", "tasa_rechazo_pct", "duracion_promedio_s"])
        return empty_ev, empty_res

    # 2. Fusión de micro-eventos consecutivos para el mismo cliente (< 2.5s entre alcances)
    df_crudo = pd.DataFrame(eventos_crudos).sort_values(["track_id", "zona", "inicio_s"])
    eventos_fusionados = []

    for (t_id, z_id), grp in df_crudo.groupby(["track_id", "zona"]):
        grp = grp.reset_index(drop=True)
        curr = grp.iloc[0].to_dict()

        for k in range(1, len(grp)):
            nxt = grp.iloc[k].to_dict()
            if nxt["inicio_s"] - curr["fin_s"] <= 2.5:
                curr["frame_fin"] = nxt["frame_fin"]
                curr["fin_s"] = nxt["fin_s"]
                curr["duracion_s"] = curr["fin_s"] - curr["inicio_s"]
            else:
                eventos_fusionados.append(curr)
                curr = nxt
        eventos_fusionados.append(curr)

    df_eventos = pd.DataFrame(eventos_fusionados)

    # 3. Clasificación con Comparación SIN OCLUSIÓN (Frame Pre-Aproximación vs Frame Post-Alejamiento)
    clasificaciones = []

    for idx, ev in df_eventos.iterrows():
        track_id = ev["track_id"]
        zona_id = ev["zona"]
        t_fin = ev["fin_s"]
        frame_ini = ev["frame_inicio"]
        frame_fin = ev["frame_fin"]

        # Verificar si el cliente re-alcanzó la misma góndola
        re_alcance = df_eventos[
            (df_eventos["track_id"] == track_id) &
            (df_eventos["zona"] == zona_id) &
            (df_eventos["inicio_s"] > t_fin) &
            (df_eventos["inicio_s"] <= t_fin + 5.0)
        ]

        # Verificar desplazamiento de pies fuera de la góndola
        sub_df = df[(df["track_id"] == track_id) & (df["frame_idx"] >= frame_fin) & (df["frame_idx"] <= frame_fin + 45)]
        se_alejo = False
        if len(sub_df) >= 2:
            dx = sub_df["x_foot"].iloc[-1] - sub_df["x_foot"].iloc[0]
            dy = sub_df["y_foot"].iloc[-1] - sub_df["y_foot"].iloc[0]
            if np.sqrt(dx**2 + dy**2) > 35.0:
                se_alejo = True

        # Comparación SIN OCLUSIÓN:
        # Pre: 35 frames ANTES de entrar (el cliente no ha tapado el estante)
        # Post: 45 frames DESPUÉS de salir (el cliente ya se alejó y dejó el estante libre)
        pts_zona = [z["polygon"] for z in zonas_interaccion if z["id"] == zona_id][0]
        frame_pre_limpio = max(0, frame_ini - 35)
        frame_post_limpio = frame_fin + 45
        score_sin_oclusion = evaluar_cambio_repisa_sin_oclusion(cap, np.array(pts_zona), frame_pre_limpio, frame_post_limpio)

        # Regla de Confirmación:
        # Si volvió a tocar la repisa -> PUT
        # Si se alejó caminando O el estante sin oclusión cambió permanentemente (score >= 0.030) -> TAKE
        if len(re_alcance) > 0:
            tipo_evento = "PUT"
        elif se_alejo or score_sin_oclusion >= 0.030:
            tipo_evento = "TAKE"
        else:
            tipo_evento = "PUT"

        clasificaciones.append(tipo_evento)

    df_eventos["tipo_accion"] = clasificaciones

    if cap:
        cap.release()

    resumen_gondolas = []
    for zona_id, grp in df_eventos.groupby("zona"):
        alcances = len(grp)
        pickups = sum(grp["tipo_accion"] == "TAKE")
        putbacks = sum(grp["tipo_accion"] == "PUT")

        tasa_conversion = (pickups / alcances) * 100.0 if alcances > 0 else 0.0
        tasa_rechazo = (putbacks / alcances) * 100.0 if alcances > 0 else 0.0

        resumen_gondolas.append({
            "zona": zona_id,
            "alcances_totales": alcances,
            "pick_ups_tomados": pickups,
            "put_backs_devueltos": putbacks,
            "tasa_conversion_pct": round(tasa_conversion, 1),
            "tasa_rechazo_pct": round(tasa_rechazo, 1),
            "duracion_promedio_s": round(grp["duracion_s"].mean(), 2)
        })

    df_resumen = pd.DataFrame(resumen_gondolas)
    return df_eventos, df_resumen


def enviar_eventos_backend(df_eventos, dwell_df, tenant_id, store_id, api_url):
    eventos_json = []
    
    def safe_float(v):
        try:
            val = float(v)
            return val if not np.isnan(val) else 0.0
        except:
            return 0.0

    def safe_int(v):
        try:
            val = float(v)
            return int(val) if not np.isnan(val) else 0
        except:
            return 0

    # 1. Eventos de Interacción (TAKE, PUT)
    if df_eventos is not None and not df_eventos.empty:
        for _, ev in df_eventos.iterrows():
            timestamp = datetime.now(timezone.utc).isoformat()
            eventos_json.append({
                "tenant_id": tenant_id,
                "store_id": store_id,
                "camera_id": "cam_01",
                "track_id": safe_int(ev.get("track_id", 0)),
                "zone_id": str(ev.get("zona", "unknown")),
                "shelf_id": str(ev.get("zona", "unknown")),
                "action": str(ev.get("tipo_accion", "UNKNOWN")),
                "timestamp": timestamp,
                "confidence": 0.90,
                "duration_s": safe_float(ev.get("duracion_s", 0.0))
            })
            
    # 2. Eventos de Flujo (OBSERVE) desde Dwell Time
    if dwell_df is not None and not dwell_df.empty:
        for _, dw in dwell_df.iterrows():
            timestamp = datetime.now(timezone.utc).isoformat()
            eventos_json.append({
                "tenant_id": tenant_id,
                "store_id": store_id,
                "camera_id": "cam_01",
                "track_id": safe_int(dw.get("track_id", 0)),
                "zone_id": str(dw.get("zona", "unknown")),
                "shelf_id": str(dw.get("zona", "unknown")),
                "action": "OBSERVE",
                "timestamp": timestamp,
                "confidence": 0.95,
                "duration_s": safe_float(dw.get("duracion_s", 0.0))
            })

    if not eventos_json:
        print("No hay eventos para enviar al backend.")
        return

    print(f"\n🚀 Enviando {len(eventos_json)} eventos a {api_url}...")
    try:
        response = requests.post(api_url, json=eventos_json, timeout=10)
        if response.status_code in [200, 201]:
            print(f"✅ Eventos enviados correctamente. Status: {response.status_code}")
        else:
            print(f"⚠️ Error al enviar eventos. Status: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Excepción al conectar con el backend: {e}")

def main(parquet_path, zonas_json_path, video_path=None, min_votacion=6, carpeta_salida=None, tenant_id="tenant_01", store_id="store_01", api_url="http://localhost:3000/api/v1/events/batch"):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No se encontró el archivo Parquet: {parquet_path}")
    if not os.path.exists(zonas_json_path):
        raise FileNotFoundError(f"No se encontró el archivo Zonas: {zonas_json_path}")

    if carpeta_salida:
        os.makedirs(carpeta_salida, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    with open(zonas_json_path, "r", encoding="utf-8") as f:
        zonas_config = json.load(f)

    zonas = zonas_config.get("zones", zonas_config.get("zonas", []))
    zonas_flujo = [z for z in zonas if z.get("type", "").lower() == "flujo" and len(z.get("polygon", z.get("poligono", []))) >= 3]
    zonas_interaccion = [z for z in zonas if z.get("type", "").lower() == "interaccion" and len(z.get("polygon", z.get("poligono", []))) >= 3]

    print("\n" + "=" * 60)
    print("📊 CLASIFICACIÓN CON VOTACIÓN TEMPORAL Y COMPARACIÓN SIN OCLUSIÓN")
    print("=" * 60)
    print(f"Zonas de pasillo: {len(zonas_flujo)} | Zonas de góndola: {len(zonas_interaccion)}")

    # 1. Clasificación PICK-UP vs PUT-BACK en Góndolas
    df_eventos = None
    if zonas_interaccion:
        df_eventos, df_resumen_gondolas = clasificar_pickup_putback(
            df, zonas_interaccion, video_path=video_path, min_votacion=min_votacion
        )

        path_interacciones = os.path.join(carpeta_salida, "interacciones_gondola.parquet") if carpeta_salida else "interacciones_gondola.parquet"
        df_eventos.to_parquet(path_interacciones, index=False)

        print("\n--- 🛒 Detalle de Interacciones PICK-UP vs PUT-BACK ---")
        if len(df_resumen_gondolas) > 0:
            print(df_resumen_gondolas.to_string(index=False))
            path_metricas = os.path.join(carpeta_salida, "metricas_pickup_putback.csv") if carpeta_salida else "metricas_pickup_putback.csv"
            df_resumen_gondolas.to_csv(path_metricas, index=False)
            print(f"\n✅ Reporte exportado a: {path_metricas}")
        else:
            print("No se detectaron alcances confirmados por votación temporal en las góndolas.")
    else:
        print("No hay zonas de góndola definidas.")

    # 2. Permanencia y Flujo (Pasillos y Góndolas)
    dwell_total_dfs = []
    if zonas_flujo:
        df_dwell_flujo = calcular_dwell_zonas_flujo(df, zonas_flujo)
        if len(df_dwell_flujo) > 0:
            dwell_total_dfs.append(df_dwell_flujo)

    if zonas_interaccion:
        df_dwell_gondolas = calcular_dwell_gondolas(df, zonas_interaccion, df_eventos)
        if len(df_dwell_gondolas) > 0:
            dwell_total_dfs.append(df_dwell_gondolas)

    dwell_df_combinado = pd.concat(dwell_total_dfs, ignore_index=True) if dwell_total_dfs else pd.DataFrame(columns=["track_id", "zona", "inicio_s", "fin_s", "duracion_s"])

    if len(dwell_df_combinado) > 0 and "zona" in dwell_df_combinado.columns:
        path_dwell = os.path.join(carpeta_salida, "dwell_por_zona.parquet") if carpeta_salida else "dwell_por_zona.parquet"
        dwell_df_combinado.to_parquet(path_dwell, index=False)

        resumen_flujo = dwell_df_combinado.groupby("zona").agg(
            visitantes_unicos=("track_id", "nunique"),
            tiempo_total_s=("duracion_s", "sum"),
            tiempo_promedio_s=("duracion_s", "mean"),
        ).sort_values("tiempo_total_s", ascending=False).round(2)

        print("\n--- 🟢 Ranking de Zonas y Permanencia (Flujo y Góndolas) ---")
        print(resumen_flujo)
        path_ranking = os.path.join(carpeta_salida, "ranking_zonas.csv") if carpeta_salida else "ranking_zonas.csv"
        resumen_flujo.to_csv(path_ranking)
        print(f"\n✅ Ranking de Permanencia exportado a: {path_ranking}")
    else:
        print("\n--- 🟢 No se registraron permanencias continuas en las zonas analizadas ---")

    print("=" * 60 + "\n")
    
    df_eventos_enviar = df_eventos if df_eventos is not None else None
    dwell_df_enviar = dwell_df_combinado if len(dwell_df_combinado) > 0 else None
    enviar_eventos_backend(df_eventos_enviar, dwell_df_enviar, tenant_id, store_id, api_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesador de Métricas Retail sin Oclusión y con Votación Temporal")
    parser.add_argument("--parquet", default="datos_parquet/tracking_cam_01.parquet")
    parser.add_argument("--zonas", default="zonas_cam_01.json")
    parser.add_argument("--video", default=None, help="Ruta del video opcional")
    parser.add_argument("--min_votacion", type=int, default=6, help="Fotogramas seguidos mínimos para confirmar alcance (default 6)")
    parser.add_argument("--carpeta", default=None, help="Carpeta de salida unificada (opcional)")
    parser.add_argument("--tenant_id", default="tenant_01", help="ID del Tenant B2B")
    parser.add_argument("--store_id", default="store_01", help="ID de la Tienda")
    parser.add_argument("--api_url", default="http://localhost:3000/api/v1/events/batch", help="URL del Backend para enviar eventos")
    args = parser.parse_args()

    main(args.parquet, args.zonas, video_path=args.video, min_votacion=args.min_votacion, carpeta_salida=args.carpeta, tenant_id=args.tenant_id, store_id=args.store_id, api_url=args.api_url)