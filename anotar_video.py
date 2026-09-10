"""
Generador de Video Anotado de Analítica Retail: Visualización completa con Zonas, Pose,
Alcance de Manos a Repisas y Renderizado Visual de PICK-UP vs PUT-BACK (Rechazo).

Guarda automáticamente en la carpeta: videos_anotados/

Uso:
  python anotar_video.py --video clip_test.mp4 --parquet datos_parquet/tracking_cam_01.parquet --zonas zonas_cam_01.json
"""

import os
import cv2
import json
import argparse
import colorsys
import importlib
import numpy as np
import pandas as pd

KEYPOINTS_CUERPO = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle",
}

ESQUELETO_CUERPO = [
    (0, 1), (0, 2),    # Nariz a ojos
    (1, 3), (2, 4),    # Ojos a orejas
    (5, 6),            # Hombro izq - Hombro der
    (5, 7), (7, 9),    # Brazo izq
    (6, 8), (8, 10),   # Brazo der
    (5, 11), (6, 12),  # Torso
    (11, 12),          # Cadera izq - Cadera der
    (11, 13), (13, 15),# Pierna izq
    (12, 14), (14, 16),# Pierna der
]


def format_mmss(segundos):
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"


def punto_en_poligono(x, y, poligono):
    if np.isnan(x) or np.isnan(y):
        return False
    return cv2.pointPolygonTest(np.array(poligono, dtype=np.float32), (float(x), float(y)), False) >= 0


def color_para_id(track_id):
    if track_id < 0:
        return (200, 200, 200)
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.9)
    return int(b * 255), int(g * 255), int(r * 255)


def anotar_video(video_path, parquet_path, zonas_path=None, carpeta_salida="videos_anotados", salida_filename=None, blur_faces=True):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No se encontró el video: {video_path}")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No se encontró el archivo Parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    os.makedirs(carpeta_salida, exist_ok=True)

    # Cargar Zonas si están disponibles
    zonas_flujo = []
    zonas_interaccion = []
    zonas_exclusion = []

    if zonas_path and os.path.exists(zonas_path):
        with open(zonas_path, "r", encoding="utf-8") as f:
            z_config = json.load(f)
        zonas_list = z_config.get("zones", z_config.get("zonas", []))
        for z in zonas_list:
            t = z.get("type", "flujo").lower()
            pts = np.array(z.get("polygon", z.get("poligono", [])), dtype=np.int32)
            if len(pts) >= 3:
                if t in ["exclusion", "exclusión", "ignorar"]:
                    zonas_exclusion.append({"id": z.get("id", "excl"), "polygon": pts})
                elif t == "interaccion":
                    zonas_interaccion.append({"id": z.get("id", "gondola"), "polygon": pts})
                else:
                    zonas_flujo.append({"id": z.get("id", "pasillo"), "polygon": pts})

    # Cargar / Ejecutar Clasificador PICK-UP vs PUT-BACK
    df_eventos = pd.DataFrame()
    try:
        procesar_mod = importlib.import_module("3_procesar_zonas")
        df_eventos, _ = procesar_mod.clasificar_pickup_putback(
            df, zonas_interaccion, video_path=video_path, min_votacion=6
        )
    except Exception as e:
        print(f"Aviso: No se pudo ejecutar clasificador de zonas ({e}), procediendo con anotación básica.")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_total_s = total_frames / fps if total_frames > 0 else 1.0

    if salida_filename is None:
        nombre_base = os.path.splitext(os.path.basename(video_path))[0]
        salida_filename = f"anotado_{nombre_base}.mp4"

    salida_path = os.path.join(carpeta_salida, salida_filename)
    out = cv2.VideoWriter(salida_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    df_por_frame = {k: v for k, v in df.groupby("frame_idx")}

    print("\n" + "=" * 60)
    print("🎥 GENERANDO VIDEO ANOTADO CON ETIQUETADO DE PICK-UP Y PUT-BACK")
    print("=" * 60)
    print(f"Video origen: {video_path} ({total_frames} frames | {format_mmss(duracion_total_s)})")
    print(f"Eventos clasificados: {len(df_eventos)} (Pick-Ups / Put-Backs)")
    print(f"Guardando video anotado en carpeta: {os.path.abspath(salida_path)}")

    # Memoria temporal para suavizado y persistencia continua de rostros
    # Dict: track_id -> {"roi": (fx1, fy1, fx2, fy2), "last_seen": frame_idx}
    estado_rostros = {}
    alpha_suavizado = 0.40  # Factor EMA de suavizado espacial
    max_persistencia_frames = int(fps * 0.75)  # Retener blur durante 0.75s si hay saltos de frames

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Dibujar Zonas de Fondo (Pasillos, Góndolas, Exclusiones)
        overlay = frame.copy()
        for z in zonas_flujo:
            cv2.polylines(frame, [z["polygon"]], True, (0, 220, 0), 2)
            cv2.putText(frame, f"Pasillo: {z['id']}", tuple(z["polygon"][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for z in zonas_interaccion:
            cv2.polylines(frame, [z["polygon"]], True, (0, 165, 255), 2)
            cv2.putText(frame, f"Gondola: {z['id']}", tuple(z["polygon"][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)

        for z in zonas_exclusion:
            cv2.polylines(frame, [z["polygon"]], True, (0, 0, 255), 2)
            cv2.fillPoly(overlay, [z["polygon"]], (30, 30, 160))

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # 2. Actualizar Coordenadas de Rostros para las Detecciones del Frame Actual
        if frame_idx in df_por_frame:
            for _, row in df_por_frame[frame_idx].iterrows():
                track_id = int(row["track_id"])
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                box_w = x2_c - x1_c
                box_h = y2_c - y1_c

                if box_w > 5 and box_h > 5:
                    # Keypoints faciales
                    nx, ny = row.get("nose_x", np.nan), row.get("nose_y", np.nan)
                    lex, ley = row.get("left_eye_x", np.nan), row.get("left_eye_y", np.nan)
                    rex, rey = row.get("right_eye_x", np.nan), row.get("right_eye_y", np.nan)
                    ear_lx, ear_ly = row.get("left_ear_x", np.nan), row.get("left_ear_y", np.nan)
                    ear_rx, ear_ry = row.get("right_ear_x", np.nan), row.get("right_ear_y", np.nan)

                    pts_face_x = [pt for pt in [nx, lex, rex, ear_lx, ear_rx] if not np.isnan(pt)]
                    pts_face_y = [pt for pt in [ny, ley, rey, ear_ly, ear_ry] if not np.isnan(pt)]

                    # Estimación robusta base (porción superior del cuerpo / hombros)
                    ls_y = row.get("left_shoulder_y", np.nan)
                    rs_y = row.get("right_shoulder_y", np.nan)
                    shoulders_y = [sy for sy in [ls_y, rs_y] if not np.isnan(sy)]

                    raw_fy1 = max(0, int(y1_c - box_h * 0.05))
                    if shoulders_y:
                        raw_fy2 = min(h, int(min(shoulders_y)))
                    else:
                        raw_fy2 = min(h, int(y1_c + box_h * 0.28))

                    cx = (x1_c + x2_c) // 2
                    half_w = int(box_w * 0.38)
                    raw_fx1 = max(0, cx - half_w)
                    raw_fx2 = min(w, cx + half_w)

                    # Si hay keypoints de rostro, expandir la caja para contenerlos completamente con margen
                    if len(pts_face_x) >= 2 and len(pts_face_y) >= 2:
                        min_kp_x, max_kp_x = min(pts_face_x), max(pts_face_x)
                        min_kp_y, max_kp_y = min(pts_face_y), max(pts_face_y)
                        kp_w = max_kp_x - min_kp_x
                        kp_h = max_kp_y - min_kp_y
                        pad_x = max(int(kp_w * 0.8), 25)
                        pad_y = max(int(kp_h * 0.8), 25)

                        raw_fx1 = min(raw_fx1, max(0, int(min_kp_x - pad_x)))
                        raw_fy1 = min(raw_fy1, max(0, int(min_kp_y - pad_y)))
                        raw_fx2 = max(raw_fx2, min(w, int(max_kp_x + pad_x)))
                        raw_fy2 = max(raw_fy2, min(h, int(max_kp_y + pad_y)))

                    raw_roi = (float(raw_fx1), float(raw_fy1), float(raw_fx2), float(raw_fy2))

                    # Suavizado exponencial (EMA) con el frame anterior para eliminar parpadeo
                    if track_id in estado_rostros:
                        prev_roi = estado_rostros[track_id]["roi"]
                        sm_fx1 = alpha_suavizado * raw_roi[0] + (1.0 - alpha_suavizado) * prev_roi[0]
                        sm_fy1 = alpha_suavizado * raw_roi[1] + (1.0 - alpha_suavizado) * prev_roi[1]
                        sm_fx2 = alpha_suavizado * raw_roi[2] + (1.0 - alpha_suavizado) * prev_roi[2]
                        sm_fy2 = alpha_suavizado * raw_roi[3] + (1.0 - alpha_suavizado) * prev_roi[3]
                        smoothed_roi = (sm_fx1, sm_fy1, sm_fx2, sm_fy2)
                    else:
                        smoothed_roi = raw_roi

                    estado_rostros[track_id] = {
                        "roi": smoothed_roi,
                        "last_seen": frame_idx
                    }

        # 3. Aplicar Blur Continuo e Ininterrumpido en Rostros (Persistencia Temporal)
        if blur_faces:
            tracks_a_eliminar = []
            for tid, info in estado_rostros.items():
                frames_desde_ultimo = frame_idx - info["last_seen"]
                if frames_desde_ultimo <= max_persistencia_frames:
                    fx1_f, fy1_f, fx2_f, fy2_f = info["roi"]
                    fx1 = max(0, int(round(fx1_f)))
                    fy1 = max(0, int(round(fy1_f)))
                    fx2 = min(w, int(round(fx2_f)))
                    fy2 = min(h, int(round(fy2_f)))

                    if (fx2 > fx1) and (fy2 > fy1):
                        roi = frame[fy1:fy2, fx1:fx2]
                        if roi.size > 0:
                            rh, rw = roi.shape[:2]
                            # Mosaico (pixelado) + Gaussian Blur
                            small_w = max(1, rw // 10)
                            small_h = max(1, rh // 10)
                            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                            pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

                            k_w = max(15, (rw // 2) * 2 + 1)
                            k_h = max(15, (rh // 2) * 2 + 1)
                            frame[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(pix, (k_w, k_h), 0)
                else:
                    tracks_a_eliminar.append(tid)

            for tid in tracks_a_eliminar:
                del estado_rostros[tid]

        # 4. Verificar Eventos Activos de PICK-UP vs PUT-BACK para el frame actual
        eventos_activos_frame = []
        if len(df_eventos) > 0 and "frame_inicio" in df_eventos.columns:
            mask_act = (df_eventos["frame_inicio"] <= frame_idx) & (frame_idx <= df_eventos["frame_fin"] + 45)
            eventos_activos_frame = df_eventos[mask_act]

        # 5. Dibujar Detecciones, Pose y Etiquetas Visuales
        if frame_idx in df_por_frame:
            for _, row in df_por_frame[frame_idx].iterrows():
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                track_id = int(row["track_id"])
                conf = float(row.get("conf", 0.0))
                color = color_para_id(track_id)

                # Bounding box del cliente
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id} ({conf:.2f})", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Punto de apoyo pies
                xf, yf = int(row["x_foot"]), int(row["y_foot"])
                cv2.circle(frame, (xf, yf), 5, color, -1)
                cv2.circle(frame, (xf, yf), 5, (255, 255, 255), 1)

                # Esqueleto corporal
                for idx1, idx2 in ESQUELETO_CUERPO:
                    k1 = KEYPOINTS_CUERPO[idx1]
                    k2 = KEYPOINTS_CUERPO[idx2]
                    if f"{k1}_x" in row and f"{k2}_x" in row:
                        x_1, y_1 = row[f"{k1}_x"], row[f"{k1}_y"]
                        x_2, y_2 = row[f"{k2}_x"], row[f"{k2}_y"]
                        c1 = row.get(f"{k1}_conf", 1.0)
                        c2 = row.get(f"{k2}_conf", 1.0)
                        if c1 > 0.3 and c2 > 0.3 and not np.isnan(x_1) and not np.isnan(x_2):
                            cv2.line(frame, (int(x_1), int(y_1)), (int(x_2), int(y_2)), color, 2)

                # Dibujar muñecas
                for mano_prefix in ["left_wrist", "right_wrist"]:
                    mx = row.get(f"{mano_prefix}_x", np.nan)
                    my = row.get(f"{mano_prefix}_y", np.nan)
                    if not np.isnan(mx) and not np.isnan(my):
                        cv2.circle(frame, (int(mx), int(my)), 4, (0, 255, 255), -1)

                # Dibujar rostro y vector de mirada únicamente si NO se activó blur
                if not blur_faces:
                    nx, ny = row.get("nose_x", np.nan), row.get("nose_y", np.nan)
                    lex, ley = row.get("left_eye_x", np.nan), row.get("left_eye_y", np.nan)
                    rex, rey = row.get("right_eye_x", np.nan), row.get("right_eye_y", np.nan)

                    if not np.isnan(nx) and not np.isnan(ny):
                        cv2.circle(frame, (int(nx), int(ny)), 3, (255, 0, 255), -1)
                        if not np.isnan(lex) and not np.isnan(rex):
                            cv2.circle(frame, (int(lex), int(ley)), 2, (0, 255, 255), -1)
                            cv2.circle(frame, (int(rex), int(rey)), 2, (0, 255, 255), -1)
                            eye_cx = (lex + rex) / 2.0
                            eye_cy = (ley + rey) / 2.0
                            gaze_vx = (nx - eye_cx) * 2.5
                            gaze_vy = (ny - eye_cy) * 2.5
                            cv2.arrowedLine(frame, (int(eye_cx), int(eye_cy)),
                                            (int(nx + gaze_vx), int(ny + gaze_vy)), (0, 230, 255), 2, tipLength=0.3)

                # Renderizar Banner de Acción (PICK-UP vs PUT-BACK) si hay evento para esta persona
                if len(eventos_activos_frame) > 0:
                    ev_persona = eventos_activos_frame[eventos_activos_frame["track_id"] == track_id]
                    if len(ev_persona) > 0:
                        tipo_act = ev_persona.iloc[0]["tipo_accion"]
                        zona_act = ev_persona.iloc[0]["zona"]

                        # Resaltar la repisa correspondiente
                        pts_z = [zg["polygon"] for zg in zonas_interaccion if zg["id"] == zona_act]

                        if tipo_act == "PICK-UP":
                            # Verde / Cyan brillante para Compra / Tomado
                            col_badge = (0, 255, 0)
                            txt_badge = f"[ID {track_id}] PICK-UP (PRODUCTO TOMADO)"
                            if pts_z:
                                cv2.polylines(frame, pts_z, True, (0, 255, 0), 3)
                        else:
                            # Rojo brillante para Devolución / Rechazo
                            col_badge = (0, 0, 255)
                            txt_badge = f"[ID {track_id}] PUT-BACK (DEVUELTO / RECHAZO)"
                            if pts_z:
                                cv2.polylines(frame, pts_z, True, (0, 0, 255), 3)

                        # Dibujar rectángulo de estado animado sobre la cabeza del cliente
                        badge_y = max(y1 - 32, 20)
                        cv2.rectangle(frame, (x1 - 5, badge_y - 18), (x1 + 330, badge_y + 6), (15, 15, 15), -1)
                        cv2.rectangle(frame, (x1 - 5, badge_y - 18), (x1 + 330, badge_y + 6), col_badge, 2)
                        cv2.putText(frame, txt_badge, (x1, badge_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col_badge, 2)

        # 4. HUD Dashboard Superior
        t_act_s = frame_idx / fps
        pct = (frame_idx / max(total_frames, 1)) * 100

        n_pickups = sum(df_eventos["tipo_accion"] == "PICK-UP") if len(df_eventos) > 0 else 0
        n_putbacks = sum(df_eventos["tipo_accion"] == "PUT-BACK") if len(df_eventos) > 0 else 0
        n_totales = len(df_eventos)
        tasa_rechazo = (n_putbacks / n_totales * 100.0) if n_totales > 0 else 0.0

        cv2.rectangle(frame, (0, 0), (w, 38), (15, 15, 15), -1)
        hud_txt = (f"ANALISIS RETAIL | Tiempo: {format_mmss(t_act_s)}/{format_mmss(duracion_total_s)} | "
                   f"PICK-UPS: {n_pickups} | PUT-BACKS (RECHAZOS): {n_putbacks} ({tasa_rechazo:.1f}%)")
        cv2.putText(frame, hud_txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 230, 255), 2)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Procesando video anotado: {frame_idx}/{total_frames} frames ({pct:.1f}%)...", end="\r")

    cap.release()
    out.release()

    print("\n" + "=" * 60)
    print(f"✅ Video anotado con PICK-UP y PUT-BACK guardado exitosamente en:")
    print(f"   📁 {os.path.abspath(salida_path)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de Video Anotado Retail con PICK-UP y PUT-BACK")
    parser.add_argument("--video", required=True, help="Ruta del video original")
    parser.add_argument("--parquet", required=True, help="Ruta del archivo .parquet de tracking")
    parser.add_argument("--zonas", default=None, help="Ruta del archivo .json de zonas")
    parser.add_argument("--carpeta", default="videos_anotados", help="Carpeta de destino para el video")
    parser.add_argument("--no-blur", action="store_true", help="Desactivar difuminado de rostros (por defecto activo para protección de datos biométricos)")
    args = parser.parse_args()

    anotar_video(args.video, args.parquet, zonas_path=args.zonas,
                 carpeta_salida=args.carpeta, salida_filename=args.salida,
                 blur_faces=not args.no_blur)