"""
Generador Avanzado de Analítica Retail: Heatmaps de Pasillos, Góndolas, Mapa Integrado, Trayectorias y Zonas.

Genera 6 mapas complementarios en la carpeta mapas_calor/:
  1. 1_mapa_calor_pasillos_permanencia.png        -> Heatmap de permanencia de pies/tronco en pasillos.
  2. 2_mapa_calor_gondolas_interaccion.png        -> Heatmap exclusivo de alcance de manos en góndolas/repisas.
  3. 3_mapa_calor_integrado_pasillos_y_gondolas.png -> NUEVO: MAPA UNIFICADO (Permanencia en Pasillos + Interacción en Góndolas).
  4. 4_trayectorias_flujo.png                       -> Rutas y trayectorias de recorrido de clientes.
  5. 5_permanencia_y_trayectorias.png               -> Mapa combinado (Permanencia + Trayectorias).
  6. 6_permanencia_por_zonas.png                     -> Mapa zonal con tarjetas de métricas por pasillo y góndola.

Uso:
  python generar_mapa_calor.py --parquet datos_parquet/tracking_cam_01.parquet --video clip_test.mp4 --zonas zonas_cam_01.json
"""

import os
import cv2
import json
import glob
import argparse
import colorsys
import numpy as np
import pandas as pd


def punto_en_poligono(x, y, poligono):
    if np.isnan(x) or np.isnan(y):
        return False
    return cv2.pointPolygonTest(np.array(poligono, dtype=np.float32), (float(x), float(y)), False) >= 0


def es_alcance_real_a_gondola(row, pts_poligono, max_dist_pies_px=220.0):
    """
    Filtro Biomecánico Anti-Ruido 2D:
    Valida proximidad de pies, extensión del brazo y orientación del torso/mirada.
    """
    lw_x, lw_y = row.get("left_wrist_x", np.nan), row.get("left_wrist_y", np.nan)
    rw_x, rw_y = row.get("right_wrist_x", np.nan), row.get("right_wrist_y", np.nan)

    lw_in = punto_en_poligono(lw_x, lw_y, pts_poligono)
    rw_in = punto_en_poligono(rw_x, rw_y, pts_poligono)

    if not (lw_in or rw_in):
        return False

    # 1. Proximidad de Pies
    xf, yf = row.get("x_foot", np.nan), row.get("y_foot", np.nan)
    dist_pies = 0.0
    if not np.isnan(xf) and not np.isnan(yf):
        dist_pies = cv2.pointPolygonTest(np.array(pts_poligono, dtype=np.float32), (float(xf), float(yf)), True)
        if dist_pies < 0 and abs(dist_pies) > max_dist_pies_px:
            return False

    # 2. Extensión del Brazo
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

    # 3. Orientación del Pecho
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


def persona_en_zona(row, pts):
    # 1. Pies
    xf, yf = row.get("x_foot", np.nan), row.get("y_foot", np.nan)
    if punto_en_poligono(xf, yf, pts):
        return True

    # 2. Centro del Tronco
    xc = row.get("x_center", np.nan)
    yc = row.get("y_center", np.nan)
    if np.isnan(xc):
        x1, x2 = row.get("x1", np.nan), row.get("x2", np.nan)
        y1, y2 = row.get("y1", np.nan), row.get("y2", np.nan)
        if not np.isnan(x1):
            xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if punto_en_poligono(xc, yc, pts):
        return True

    # 3. Caderas
    lh_x, lh_y = row.get("left_hip_x", np.nan), row.get("left_hip_y", np.nan)
    rh_x, rh_y = row.get("right_hip_x", np.nan), row.get("right_hip_y", np.nan)
    if not np.isnan(lh_x) and not np.isnan(rh_x):
        if punto_en_poligono((lh_x + rh_x) / 2.0, (lh_y + rh_y) / 2.0, pts):
            return True

    return False


def obtener_posicion_representativa(row):
    xf, yf = row.get("x_foot", np.nan), row.get("y_foot", np.nan)
    if not np.isnan(xf) and not np.isnan(yf):
        return int(xf), int(yf)

    lh_x, lh_y = row.get("left_hip_x", np.nan), row.get("left_hip_y", np.nan)
    rh_x, rh_y = row.get("right_hip_x", np.nan), row.get("right_hip_y", np.nan)
    if not np.isnan(lh_x) and not np.isnan(rh_x):
        return int((lh_x + rh_x) / 2.0), int((lh_y + rh_y) / 2.0)

    xc, yc = row.get("x_center", np.nan), row.get("y_center", np.nan)
    if not np.isnan(xc) and not np.isnan(yc):
        return int(xc), int(yc)

    x1, x2 = row.get("x1", np.nan), row.get("x2", np.nan)
    y1, y2 = row.get("y1", np.nan), row.get("y2", np.nan)
    if not np.isnan(x1):
        return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)

    return 0, 0


def color_para_id(track_id):
    if track_id < 0:
        return (200, 200, 200)
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def obtener_color_permanencia(intensidad_0_1):
    val = int(np.clip(intensidad_0_1, 0, 1) * 255)
    img_val = np.uint8([[val]])
    color_map = cv2.applyColorMap(img_val, cv2.COLORMAP_JET)
    b, g, r = color_map[0, 0]
    return int(b), int(g), int(r)


def cargar_zonas_y_exclusiones(zonas_json_path):
    if not zonas_json_path or not os.path.exists(zonas_json_path):
        return [], []
    with open(zonas_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    zonas = config.get("zones", config.get("zonas", []))
    zonas_validas = []
    zonas_exclusion = []

    for z in zonas:
        t = z.get("type", "flujo").lower()
        pts = np.array(z.get("polygon", z.get("poligono", [])), dtype=np.int32)
        if len(pts) < 3:
            continue
        if t in ["exclusion", "exclusión", "ignorar"]:
            zonas_exclusion.append({"id": z.get("id", "exclusion"), "polygon": pts})
        else:
            zonas_validas.append({"id": z.get("id", "zona"), "type": t, "polygon": pts})

    return zonas_validas, zonas_exclusion


def filtrar_puntos_por_exclusion(df, zonas_exclusion):
    if not zonas_exclusion or len(df) == 0:
        return df

    mask_excluir = np.zeros(len(df), dtype=bool)

    for idx, (_, row) in enumerate(df.iterrows()):
        xf, yf = row["x_foot"], row["y_foot"]
        xc, yc = row.get("x_center", np.nan), row.get("y_center", np.nan)
        lw_x, lw_y = row.get("left_wrist_x", np.nan), row.get("left_wrist_y", np.nan)
        rw_x, rw_y = row.get("right_wrist_x", np.nan), row.get("right_wrist_y", np.nan)

        for ze in zonas_exclusion:
            pts = ze["polygon"]
            if (punto_en_poligono(xf, yf, pts) or
                punto_en_poligono(xc, yc, pts) or
                punto_en_poligono(lw_x, lw_y, pts) or
                punto_en_poligono(rw_x, rw_y, pts)):
                mask_excluir[idx] = True
                break

    n_excluidas = np.sum(mask_excluir)
    if n_excluidas > 0:
        print(f"  🚫 Filtro de Exclusión Activo: Se eliminaron {n_excluidas} detecciones en áreas de exclusión.")

    return df[~mask_excluir].copy()


def calcular_pesos_trayectoria_permanencia(df, min_std_kp=0.8, window_sec=1.0):
    df = df.dropna(subset=["x_foot", "y_foot"]).copy()
    filas_filtradas = []

    kp_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]

    for track_id, grupo in df.groupby("track_id"):
        grupo = grupo.sort_values("frame_idx")
        if len(grupo) < 5:
            continue

        dx = grupo["x_foot"].diff().fillna(0)
        dy = grupo["y_foot"].diff().fillna(0)
        dt = grupo["timestamp_s"].diff().fillna(0.04)

        vel_px_s = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 0.001)
        pesos_vel = 1.0 / (1.0 + (vel_px_s / 20.0)**2)

        fps_est = 25.0
        window_frames = int(round(fps_est * window_sec))

        if kp_cols and len(grupo) >= 3:
            roll_std = grupo[kp_cols].rolling(window=window_frames, min_periods=3).std()
            mean_std_cuerpo = roll_std.mean(axis=1).fillna(10.0)
            filtro_humano = np.where(mean_std_cuerpo >= min_std_kp, 1.0, 0.0)
        else:
            filtro_humano = 1.0

        grupo["peso_permanencia"] = pesos_vel * filtro_humano
        filas_filtradas.append(grupo)

    if filas_filtradas:
        res = pd.concat(filas_filtradas, ignore_index=True)
        res_val = res[res["peso_permanencia"] > 0.01].copy()
        if len(res_val) > 0:
            return res_val
        return res
    return pd.DataFrame()


def aplicar_cero_absoluto_exclusion(img, zonas_exclusion):
    if not zonas_exclusion:
        return img
    res = img.copy()
    for ze in zonas_exclusion:
        pts = ze["polygon"]
        cv2.fillPoly(res, [pts], 0)
    return res


def dibujar_zonas_exclusion_sobre_imagen(img, zonas_exclusion):
    resultado = img.copy()
    overlay = img.copy()

    for ze in zonas_exclusion:
        pts = ze["polygon"]
        cv2.fillPoly(overlay, [pts], (40, 40, 180))
        cv2.polylines(resultado, [pts], True, (0, 0, 255), 2)

        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        cv2.putText(resultado, f"[EXCLUIDO: {ze['id']}]", (x_min + 5, y_min + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    resultado = cv2.addWeighted(overlay, 0.25, resultado, 0.75, 0)
    return resultado


# ============================================================
# 1. MAPA DE PASILLOS (PERMANENCIA EN PISO/TRONCO)
# ============================================================
def generar_mapa_permanencia_continuo(df_val, fondo, zonas_exclusion, salida_png, sigma=30):
    h, w = fondo.shape[:2]
    heatmap_acc = np.zeros((h, w), dtype=np.float32)

    for _, row in df_val.iterrows():
        xi, yi = obtener_posicion_representativa(row)
        peso = float(row.get("peso_permanencia", 1.0))
        if 0 <= xi < w and 0 <= yi < h:
            heatmap_acc[yi, xi] += peso

    heatmap_acc = aplicar_cero_absoluto_exclusion(heatmap_acc, zonas_exclusion)
    suavizado = cv2.GaussianBlur(heatmap_acc, (0, 0), sigmaX=sigma, sigmaY=sigma)
    suavizado = aplicar_cero_absoluto_exclusion(suavizado, zonas_exclusion)

    resultado = fondo.copy()

    if suavizado.max() > 0:
        pct_high = np.percentile(suavizado[suavizado > 0], 97) if np.any(suavizado > 0) else 1.0
        norm_pow = np.power(suavizado / max(pct_high, 0.001), 0.45)
        normalizado = np.clip(norm_pow * 255.0, 0, 255).astype(np.uint8)
        normalizado = aplicar_cero_absoluto_exclusion(normalizado, zonas_exclusion)

        color = cv2.applyColorMap(normalizado, cv2.COLORMAP_JET)
        mascara = normalizado > 15
        mezcla = cv2.addWeighted(fondo, 0.30, color, 0.70, 0)
        resultado[mascara] = mezcla[mascara]

    if zonas_exclusion:
        resultado = dibujar_zonas_exclusion_sobre_imagen(resultado, zonas_exclusion)

    n_personas = df_val["track_id"].nunique()
    cv2.rectangle(resultado, (0, 0), (w, 38), (15, 15, 15), -1)
    cv2.putText(resultado, f"1. MAPA DE PASILLOS (PERMANENCIA) | {n_personas} personas",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

    cv2.imwrite(salida_png, resultado)
    print(f"  [1/6] Mapa de pasillos guardado en: {salida_png}")
    return resultado


# ============================================================
# 2. MAPA EXCLUSIVO DE GÓNDOLAS (ALCANCE DE MANOS Y PRODUCTOS)
# ============================================================
def generar_mapa_gondolas_exclusivo(df_val, fondo, zonas_validas, zonas_exclusion, salida_png, sigma=20):
    h, w = fondo.shape[:2]
    heatmap_gondola = np.zeros((h, w), dtype=np.float32)

    zonas_gondola = [z for z in zonas_validas if z.get("type", "").lower() == "interaccion"]
    puntos_contacto_manos = []

    tiene_manos = "left_wrist_x" in df_val.columns and "right_wrist_x" in df_val.columns

    if tiene_manos:
        for _, row in df_val.iterrows():
            peso = float(row.get("peso_permanencia", 1.0))
            for zg in zonas_gondola:
                if es_alcance_real_a_gondola(row, zg["polygon"]):
                    for wrist_x, wrist_y in [("left_wrist_x", "left_wrist_y"), ("right_wrist_x", "right_wrist_y")]:
                        wx, wy = row[wrist_x], row[wrist_y]
                        if not np.isnan(wx) and not np.isnan(wy):
                            wxi, wyi = int(wx), int(wy)
                            if 0 <= wxi < w and 0 <= wyi < h and punto_en_poligono(wxi, wyi, zg["polygon"]):
                                heatmap_gondola[wyi, wxi] += peso * 2.0
                                puntos_contacto_manos.append((wxi, wyi))

    heatmap_gondola = aplicar_cero_absoluto_exclusion(heatmap_gondola, zonas_exclusion)
    suavizado = cv2.GaussianBlur(heatmap_gondola, (0, 0), sigmaX=sigma, sigmaY=sigma)
    suavizado = aplicar_cero_absoluto_exclusion(suavizado, zonas_exclusion)

    resultado = fondo.copy()

    for zg in zonas_gondola:
        pts = zg["polygon"]
        cv2.polylines(resultado, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
        cv2.putText(resultado, f"Gondola: {zg['id']}", tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 2)

    if suavizado.max() > 0:
        pct_high = np.percentile(suavizado[suavizado > 0], 97) if np.any(suavizado > 0) else 1.0
        norm_pow = np.power(suavizado / max(pct_high, 0.001), 0.45)
        normalizado = np.clip(norm_pow * 255.0, 0, 255).astype(np.uint8)
        normalizado = aplicar_cero_absoluto_exclusion(normalizado, zonas_exclusion)

        color = cv2.applyColorMap(normalizado, cv2.COLORMAP_JET)
        mascara = normalizado > 12
        mezcla = cv2.addWeighted(fondo, 0.25, color, 0.75, 0)
        resultado[mascara] = mezcla[mascara]

    for pt in puntos_contacto_manos:
        cv2.circle(resultado, pt, 4, (255, 0, 255), -1)
        cv2.circle(resultado, pt, 5, (255, 255, 255), 1)

    if zonas_exclusion:
        resultado = dibujar_zonas_exclusion_sobre_imagen(resultado, zonas_exclusion)

    n_alcances = len(puntos_contacto_manos)
    cv2.rectangle(resultado, (0, 0), (w, 38), (15, 15, 15), -1)
    cv2.putText(resultado, f"2. MAPA EXCLUSIVO DE GONDOLAS (MANOS) | {n_alcances} contactos",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

    cv2.imwrite(salida_png, resultado)
    print(f"  [2/6] Mapa exclusivo de góndolas guardado en: {salida_png}")
    return resultado


# ============================================================
# 3. NUEVO: MAPA DE CALOR INTEGRADO (PASILLOS + GÓNDOLAS)
# ============================================================
def generar_mapa_integrado_pasillos_y_gondolas(df_val, fondo, zonas_validas, zonas_exclusion, salida_png, sigma=25):
    h, w = fondo.shape[:2]
    heatmap_acc = np.zeros((h, w), dtype=np.float32)

    # 1. Acumular permanencia de cuerpo en pasillos
    for _, row in df_val.iterrows():
        xi, yi = obtener_posicion_representativa(row)
        peso = float(row.get("peso_permanencia", 1.0))
        if 0 <= xi < w and 0 <= yi < h:
            heatmap_acc[yi, xi] += peso

    # 2. Acumular interacción de manos en góndolas
    zonas_gondola = [z for z in zonas_validas if z.get("type", "").lower() == "interaccion"]
    puntos_contacto_manos = []
    tiene_manos = "left_wrist_x" in df_val.columns and "right_wrist_x" in df_val.columns

    if tiene_manos:
        for _, row in df_val.iterrows():
            peso = float(row.get("peso_permanencia", 1.0))
            for zg in zonas_gondola:
                if es_alcance_real_a_gondola(row, zg["polygon"]):
                    for wrist_x, wrist_y in [("left_wrist_x", "left_wrist_y"), ("right_wrist_x", "right_wrist_y")]:
                        wx, wy = row[wrist_x], row[wrist_y]
                        if not np.isnan(wx) and not np.isnan(wy):
                            wxi, wyi = int(wx), int(wy)
                            if 0 <= wxi < w and 0 <= wyi < h and punto_en_poligono(wxi, wyi, zg["polygon"]):
                                heatmap_acc[wyi, wxi] += peso * 2.5
                                puntos_contacto_manos.append((wxi, wyi))

    heatmap_acc = aplicar_cero_absoluto_exclusion(heatmap_acc, zonas_exclusion)
    suavizado = cv2.GaussianBlur(heatmap_acc, (0, 0), sigmaX=sigma, sigmaY=sigma)
    suavizado = aplicar_cero_absoluto_exclusion(suavizado, zonas_exclusion)

    resultado = fondo.copy()

    for zg in zonas_gondola:
        pts = zg["polygon"]
        cv2.polylines(resultado, [pts], isClosed=True, color=(0, 165, 255), thickness=2)

    if suavizado.max() > 0:
        pct_high = np.percentile(suavizado[suavizado > 0], 97) if np.any(suavizado > 0) else 1.0
        norm_pow = np.power(suavizado / max(pct_high, 0.001), 0.45)
        normalizado = np.clip(norm_pow * 255.0, 0, 255).astype(np.uint8)
        normalizado = aplicar_cero_absoluto_exclusion(normalizado, zonas_exclusion)

        color = cv2.applyColorMap(normalizado, cv2.COLORMAP_JET)
        mascara = normalizado > 15
        mezcla = cv2.addWeighted(fondo, 0.28, color, 0.72, 0)
        resultado[mascara] = mezcla[mascara]

    for pt in puntos_contacto_manos:
        cv2.circle(resultado, pt, 4, (255, 0, 255), -1)
        cv2.circle(resultado, pt, 5, (255, 255, 255), 1)

    if zonas_exclusion:
        resultado = dibujar_zonas_exclusion_sobre_imagen(resultado, zonas_exclusion)

    n_personas = df_val["track_id"].nunique()
    cv2.rectangle(resultado, (0, 0), (w, 38), (15, 15, 15), -1)
    cv2.putText(resultado, f"3. MAPA UNIFICADO (PASILLOS + GONDOLAS) | {n_personas} personas | {len(puntos_contacto_manos)} alcances",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

    cv2.imwrite(salida_png, resultado)
    print(f"  [3/6] Mapa unificado de pasillos + góndolas guardado en: {salida_png}")
    return resultado


# ============================================================
# 4. MAPA DE TRAYECTORIAS Y RUTAS
# ============================================================
def generar_mapa_trayectorias(df_val, fondo, zonas_exclusion, salida_png):
    resultado = fondo.copy()
    n_personas = df_val["track_id"].nunique()

    for track_id, grupo in df_val.groupby("track_id"):
        grupo = grupo.sort_values("frame_idx")
        pts_list = []
        for _, row in grupo.iterrows():
            xi, yi = obtener_posicion_representativa(row)
            if xi > 0 and yi > 0:
                pts_list.append([xi, yi])

        pts = np.array(pts_list, dtype=np.int32)

        if len(pts) >= 2:
            color = color_para_id(int(track_id))
            cv2.polylines(resultado, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(resultado, tuple(pts[0]), 3, (255, 255, 255), -1)
            cv2.circle(resultado, tuple(pts[-1]), 5, color, -1)
            cv2.circle(resultado, tuple(pts[-1]), 5, (255, 255, 255), 1)

    if zonas_exclusion:
        resultado = dibujar_zonas_exclusion_sobre_imagen(resultado, zonas_exclusion)

    cv2.rectangle(resultado, (0, 0), (w := fondo.shape[1], 38), (15, 15, 15), -1)
    cv2.putText(resultado, f"4. TRAYECTORIAS DE RECORRIDO | {n_personas} clientes",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

    cv2.imwrite(salida_png, resultado)
    print(f"  [4/6] Mapa de trayectorias guardado en: {salida_png}")
    return resultado


# ============================================================
# 5. MAPA COMBINADO: PERMANENCIA + TRAYECTORIAS
# ============================================================
def generar_mapa_combinado(df_val, img_heatmap, zonas_exclusion, salida_png):
    resultado = img_heatmap.copy()
    n_personas = df_val["track_id"].nunique()

    for track_id, grupo in df_val.groupby("track_id"):
        grupo = grupo.sort_values("frame_idx")
        pts_list = []
        for _, row in grupo.iterrows():
            xi, yi = obtener_posicion_representativa(row)
            if xi > 0 and yi > 0:
                pts_list.append([xi, yi])

        pts = np.array(pts_list, dtype=np.int32)

        if len(pts) >= 2:
            color = color_para_id(int(track_id))
            cv2.polylines(resultado, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(resultado, tuple(pts[-1]), 4, color, -1)

    if zonas_exclusion:
        resultado = dibujar_zonas_exclusion_sobre_imagen(resultado, zonas_exclusion)

    cv2.rectangle(resultado, (0, 0), (w := resultado.shape[1], 38), (15, 15, 15), -1)
    cv2.putText(resultado, f"5. PERMANENCIA + TRAYECTORIAS COMBINADAS | {n_personas} personas",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

    cv2.imwrite(salida_png, resultado)
    print(f"  [5/6] Mapa combinado guardado en: {salida_png}")


# ============================================================
# 6. PERMANENCIA E INTERACCIÓN DE MANOS POR GÓNDOLA / ZONA
# ============================================================
def generar_mapa_zonas(df_val, fondo, zonas_validas, zonas_exclusion, salida_png):
    resultado = fondo.copy()
    overlay = fondo.copy()
    h, w = fondo.shape[:2]

    if not zonas_validas:
        print("  [6/6] No se encontraron zonas analizables.")
        return

    fps = 25.0
    if len(df_val) > 1:
        diffs = df_val["frame_idx"].diff().dropna()
        frame_step = diffs.median() if len(diffs) > 0 and diffs.median() > 0 else 1
    else:
        frame_step = 1

    seg_por_registro = frame_step / fps
    tiene_manos = "left_wrist_x" in df_val.columns and "right_wrist_x" in df_val.columns

    stats_zonas = []
    puntos_interaccion_manos = []

    for z in zonas_validas:
        nombre = z["id"]
        pts = z["polygon"]

        mask_presencia_cuerpo = [persona_en_zona(row, pts) for _, row in df_val.iterrows()]
        mask_manos = [False] * len(df_val)

        if tiene_manos and z.get("type", "").lower() == "interaccion":
            for idx, (_, row) in enumerate(df_val.iterrows()):
                if es_alcance_real_a_gondola(row, pts):
                    lw_in = punto_en_poligono(row["left_wrist_x"], row["left_wrist_y"], pts)
                    rw_in = punto_en_poligono(row["right_wrist_x"], row["right_wrist_y"], pts)

                    if lw_in:
                        puntos_interaccion_manos.append((int(row["left_wrist_x"]), int(row["left_wrist_y"])))
                    if rw_in:
                        puntos_interaccion_manos.append((int(row["right_wrist_x"]), int(row["right_wrist_y"])))

                    mask_manos[idx] = lw_in or rw_in

        mask_presencia = [p or m for p, m in zip(mask_presencia_cuerpo, mask_manos)]
        df_z = df_val[mask_presencia]

        n_reg = len(df_z)
        t_total = n_reg * seg_por_registro
        vis_unicos = df_z["track_id"].nunique() if n_reg > 0 else 0
        n_alcances_mano = sum(mask_manos)

        stats_zonas.append({
            "nombre": nombre, "polygon": pts,
            "t_total": t_total, "vis_unicos": vis_unicos,
            "alcances_mano": n_alcances_mano
        })

    max_t = max([s["t_total"] for s in stats_zonas], default=1.0) or 1.0

    for s in stats_zonas:
        pts = s["polygon"]
        intensidad = s["t_total"] / max_t
        color_bgr = obtener_color_permanencia(intensidad)
        cv2.fillPoly(overlay, [pts], color_bgr)

    resultado = cv2.addWeighted(overlay, 0.45, fondo, 0.55, 0)

    for pt in puntos_interaccion_manos:
        cv2.circle(resultado, pt, 5, (255, 0, 255), -1)
        cv2.circle(resultado, pt, 6, (255, 255, 255), 1)

    for idx_z, s in enumerate(stats_zonas):
        pts = s["polygon"]
        intensidad = s["t_total"] / max_t
        color_bgr = obtener_color_permanencia(intensidad)

        cv2.polylines(resultado, [pts], True, (255, 255, 255), 2)

        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))

        txt1 = f"Zona: {s['nombre']}"
        txt2 = f"Perm: {s['t_total']:.1f}s | {s['vis_unicos']} pers"
        txt3 = f"Manos en Gondola: {s['alcances_mano']} alcances" if s["alcances_mano"] > 0 else "Manos: 0 alcances"

        box_w, box_h = 230, 56
        box_x = max(min(x_min, w - box_w - 10), 10)
        box_y = max(y_min - box_h - (idx_z % 2) * 60 - 5, 10)

        sub_img = resultado[box_y:box_y+box_h, box_x:box_x+box_w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res_box = cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 0)
        resultado[box_y:box_y+box_h, box_x:box_x+box_w] = res_box
        cv2.rectangle(resultado, (box_x, box_y), (box_x + box_w, box_y + box_h), color_bgr, 2)

        cv2.putText(resultado, txt1, (box_x + 6, box_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(resultado, txt2, (box_x + 6, box_y + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1)
        cv2.putText(resultado, txt3, (box_x + 6, box_y + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 120, 255), 1)

    if zonas_exclusion:
        resultado = dibujar_zonas_exclusion_sobre_imagen(resultado, zonas_exclusion)

    cv2.rectangle(resultado, (0, 0), (w, 38), (15, 15, 15), -1)
    cv2.putText(resultado, f"6. PERMANENCIA E INTERACCION DE MANOS | {len(stats_zonas)} zonas analizadas",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

    cv2.imwrite(salida_png, resultado)
    print(f"  [6/6] Mapa de interacción por zonas guardado en: {salida_png}")


def main(parquet_path, video_path, zonas_path=None, carpeta_salida="mapas_calor"):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No se encontró el Parquet: {parquet_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    ret, fondo = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("No se pudo leer la imagen del video")

    df = pd.read_parquet(parquet_path)
    os.makedirs(carpeta_salida, exist_ok=True)

    if zonas_path is None or not os.path.exists(zonas_path):
        candidatos = glob.glob("zonas*.json") + glob.glob("zones*.json")
        if candidatos:
            zonas_path = candidatos[0]

    zonas_validas, zonas_exclusion = cargar_zonas_y_exclusiones(zonas_path)

    print("\n" + "=" * 60)
    print("🚀 GENERANDO SUITE AVANZADA DE MAPAS INTEGRADOS")
    print("=" * 60)

    # 1. Filtro de exclusión
    df_filtrado = filtrar_puntos_por_exclusion(df, zonas_exclusion)

    # 2. Filtro biológico de micro-movimiento
    df_val = calcular_pesos_trayectoria_permanencia(df_filtrado, min_std_kp=0.8, window_sec=1.0)

    if len(df_val) == 0:
        print("⚠️ No hay trayectorias humanas válidas fuera de las zonas de exclusión.")
        return

    # 1. Mapa de Pasillos (Permanencia)
    generar_mapa_permanencia_continuo(
        df_val, fondo, zonas_exclusion, os.path.join(carpeta_salida, "1_mapa_calor_pasillos_permanencia.png")
    )

    # 2. Mapa EXCLUSIVO de Góndolas (Manos)
    generar_mapa_gondolas_exclusivo(
        df_val, fondo, zonas_validas, zonas_exclusion, os.path.join(carpeta_salida, "2_mapa_calor_gondolas_interaccion.png")
    )

    # 3. NUEVO: MAPA INTEGRADO UNIFICADO (Pasillos + Góndolas)
    img_integrado = generar_mapa_integrado_pasillos_y_gondolas(
        df_val, fondo, zonas_validas, zonas_exclusion, os.path.join(carpeta_salida, "3_mapa_calor_integrado_pasillos_y_gondolas.png")
    )

    # 4. Mapa de Trayectorias
    generar_mapa_trayectorias(df_val, fondo, zonas_exclusion, os.path.join(carpeta_salida, "4_trayectorias_flujo.png"))

    # 5. Mapa Combinado (Heatmap Integrado + Trayectorias)
    generar_mapa_combinado(df_val, img_integrado, zonas_exclusion, os.path.join(carpeta_salida, "5_permanencia_y_trayectorias.png"))

    # 6. Mapa Zonal
    if zonas_validas:
        generar_mapa_zonas(df_val, fondo, zonas_validas, zonas_exclusion, os.path.join(carpeta_salida, "6_permanencia_por_zonas.png"))

    print("=" * 60)
    print(f"✅ Todos los 6 mapas se guardaron en la carpeta: {os.path.abspath(carpeta_salida)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suite de Analítica Retail Integrada")
    parser.add_argument("--parquet", default="datos_parquet/tracking_cam_01.parquet")
    parser.add_argument("--video", default="clip_test.mp4")
    parser.add_argument("--zonas", default=None)
    parser.add_argument("--carpeta", default="mapas_calor")
    args = parser.parse_args()

    main(args.parquet, args.video, args.zonas, carpeta_salida=args.carpeta)
