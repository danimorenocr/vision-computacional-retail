"""
Generador de Video Anotado Retail con Blur YuNet (Sin Frame Skip, Primeros 15 Segundos).

Procesa cuadro a cuadro (frame_skip=1) los primeros 15 segundos de un video,
aplicando la red neuronal YuNet (cv2.FaceDetectorYN) para detección y difuminado
elíptico continuo de rostros (cumplimiento Ley 1581 de protección de datos).

Uso:
  python anotar_video_yunet_15s.py --video clip_test.mp4 --parquet datos_parquet/tracking_clip_test.parquet
  python anotar_video_yunet_15s.py --video clip_test.mp4 --max-segundos 15
"""

import os
import cv2
import json
import argparse
import colorsys
import importlib
import urllib.request
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
    ms = int((segundos % 1) * 10)
    return f"{m:02d}:{s:02d}.{ms}"


def color_para_id(track_id):
    if track_id < 0:
        return (200, 200, 200)
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.9)
    return int(b * 255), int(g * 255), int(r * 255)


def obtener_detector_yunet(model_path="face_detection_yunet_2023mar.onnx"):
    """
    Inicializa el detector YuNet de OpenCV (cv2.FaceDetectorYN).
    Descarga automáticamente el modelo .onnx si no existe localmente.
    """
    if not os.path.exists(model_path):
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        print(f"Descargando modelo YuNet desde {url}...")
        try:
            urllib.request.urlretrieve(url, model_path)
            print("[OK] Modelo YuNet descargado con exito.")
        except Exception as e:
            print(f"[WARN] Error al descargar YuNet: {e}")
            return None

    try:
        detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(300, 300),
            score_threshold=0.20,
            nms_threshold=0.3,
            top_k=5000
        )
        return detector
    except Exception as e:
        print(f"[WARN] No se pudo inicializar cv2.FaceDetectorYN: {e}")
        return None


def buscar_o_generar_parquet(video_path, parquet_path=None):
    """
    Busca un archivo Parquet existente o ejecuta la extracción con frame_skip=1 (sin salto de cuadros).
    """
    if parquet_path and os.path.exists(parquet_path):
        return parquet_path

    nombre_base = os.path.splitext(os.path.basename(video_path))[0]
    candidatos = [
        os.path.join("datos_parquet", f"tracking_{nombre_base}.parquet"),
        os.path.join("datos_parquet", f"{nombre_base}.parquet"),
        os.path.join("datos_parquet", f"tracking_cam_01.parquet")
    ]
    for c in candidatos:
        if os.path.exists(c):
            print(f"[INFO] Usando archivo Parquet encontrado: {c}")
            return c

    print(f"[INIT] No se proporciono Parquet. Ejecutando extraccion sin frame skip (frame_skip=1)...")
    try:
        mod_extraer = importlib.import_module("1_extraer_datos")
        salida_p = os.path.join("datos_parquet", f"tracking_{nombre_base}_fskip1.parquet")
        df_res = mod_extraer.modo_guardar(
            video_path=video_path,
            camera_id="cam_01",
            frame_skip=1,
            salida_parquet=salida_p
        )
        return salida_p
    except Exception as e:
        print(f"[WARN] Error al extraer datos tracking: {e}")
        return None


def anotar_video_yunet_15s(
    video_path,
    parquet_path=None,
    zonas_path=None,
    carpeta_salida="videos_anotados",
    salida_filename=None,
    max_segundos=15.0,
    blur_faces=True,
    yunet_model_path="face_detection_yunet_2023mar.onnx"
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    parquet_valido = buscar_o_generar_parquet(video_path, parquet_path)
    if not parquet_valido or not os.path.exists(parquet_valido):
        raise FileNotFoundError(f"No se encontró ni se pudo generar el archivo Parquet para: {video_path}")

    df = pd.read_parquet(parquet_valido)
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

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Limitar strictly a los primeros max_segundos (default 15s)
    max_frames = int(round(fps * max_segundos))
    frames_a_procesar = min(total_frames_video, max_frames)
    duracion_proc_s = frames_a_procesar / fps

    if salida_filename is None:
        nombre_base = os.path.splitext(os.path.basename(video_path))[0]
        salida_filename = f"anotado_15s_yunet_{nombre_base}.mp4"

    salida_path = os.path.join(carpeta_salida, salida_filename)
    out = cv2.VideoWriter(salida_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    df_filtrado = df[df["frame_idx"] < max_frames]
    df_por_frame = {k: v for k, v in df_filtrado.groupby("frame_idx")}

    print("\n" + "=" * 65)
    print("[VIDEO] PROCESANDO VIDEO (PRIMEROS 15 SEGUNDOS | FRAME SKIP = 1)")
    print("=" * 65)
    print(f"Video origen: {video_path}")
    print(f"Procesando cuadros: 0 a {frames_a_procesar} ({duracion_proc_s:.1f}s a {fps:.1f} FPS sin salto de frames)")
    print(f"Guardando en: {os.path.abspath(salida_path)}")

    # Detector de Rostros YuNet (cv2.FaceDetectorYN)
    yunet_detector = obtener_detector_yunet(yunet_model_path) if blur_faces else None
    if blur_faces and yunet_detector is not None:
        print("[FACE] Detector YuNet (cv2.FaceDetectorYN) activado para difuminado de rostros.")

    # Memoria temporal para suavizado espacial (EMA) y continuidad sin parpadeo
    # Dict: track_id -> {"roi": (fx1, fy1, fx2, fy2), "last_seen": frame_idx}
    estado_rostros = {}
    alpha_suavizado = 0.65  # Factor EMA más reactivo para acompañar movimiento sin rezagarse
    max_persistencia_frames = int(fps * 0.85)  # Persistencia de blur por 0.85s

    frame_idx = 0

    while frame_idx < frames_a_procesar:
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

        # 2. Detección de Rostros con YuNet cuadro a cuadro (frame_skip=1)
        if frame_idx in df_por_frame:
            for _, row in df_por_frame[frame_idx].iterrows():
                track_id = int(row["track_id"])
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                box_w = x2_c - x1_c
                box_h = y2_c - y1_c

                if box_w > 5 and box_h > 5:
                    raw_fx1, raw_fy1, raw_fx2, raw_fy2 = None, None, None, None

                    # Intento 1: Detección exacta con YuNet en hasta 70% de la altura del cuerpo
                    if yunet_detector is not None:
                        crop_y2 = min(h, y1_c + int(box_h * 0.70))
                        crop = frame[y1_c:crop_y2, x1_c:x2_c]
                        ch, cw = crop.shape[:2]
                        if cw >= 10 and ch >= 10:
                            yunet_detector.setInputSize((cw, ch))
                            _, faces = yunet_detector.detect(crop)
                            if faces is not None and len(faces) > 0:
                                caras_validas = [f for f in faces if f[-1] >= 0.20]
                                if caras_validas:
                                    best_face = max(caras_validas, key=lambda f: f[-1])
                                    fx, fy, fw, fh = best_face[0:4]
                                    pad_w = int(fw * 0.50)
                                    pad_h = int(fh * 0.55)
                                    raw_fx1 = max(0, x1_c + int(fx) - pad_w)
                                    raw_fy1 = max(0, y1_c + int(fy) - pad_h)
                                    raw_fx2 = min(w, x1_c + int(fx + fw) + pad_w)
                                    raw_fy2 = min(h, y1_c + int(fy + fh) + pad_h)

                    # Intento 2: Fallback proporcional / keypoints cuando YuNet no detecta (de espaldas/perfil)
                    if raw_fx1 is None:
                        nx, ny = row.get("nose_x", np.nan), row.get("nose_y", np.nan)
                        lex, ley = row.get("left_eye_x", np.nan), row.get("left_eye_y", np.nan)
                        rex, rey = row.get("right_eye_x", np.nan), row.get("right_eye_y", np.nan)
                        ear_lx, ear_ly = row.get("left_ear_x", np.nan), row.get("left_ear_y", np.nan)
                        ear_rx, ear_ry = row.get("right_ear_x", np.nan), row.get("right_ear_y", np.nan)

                        pts_face_x = [pt for pt in [nx, lex, rex, ear_lx, ear_rx] if not np.isnan(pt)]
                        pts_face_y = [pt for pt in [ny, ley, rey, ear_ly, ear_ry] if not np.isnan(pt)]

                        ls_y = row.get("left_shoulder_y", np.nan)
                        rs_y = row.get("right_shoulder_y", np.nan)
                        shoulders_y = [sy for sy in [ls_y, rs_y] if not np.isnan(sy)]

                        raw_fy1 = max(0, int(y1_c - box_h * 0.08))
                        if shoulders_y:
                            raw_fy2 = min(h, int(min(shoulders_y)) + int(box_h * 0.05))
                        else:
                            raw_fy2 = min(h, int(y1_c + box_h * 0.35))

                        cx = (x1_c + x2_c) // 2
                        half_w = int(box_w * 0.45)
                        raw_fx1 = max(0, cx - half_w)
                        raw_fx2 = min(w, cx + half_w)

                        if len(pts_face_x) >= 2 and len(pts_face_y) >= 2:
                            min_kp_x, max_kp_x = min(pts_face_x), max(pts_face_x)
                            min_kp_y, max_kp_y = min(pts_face_y), max(pts_face_y)
                            kp_w = max_kp_x - min_kp_x
                            kp_h = max_kp_y - min_kp_y
                            pad_x = max(int(kp_w * 0.9), 30)
                            pad_y = max(int(kp_h * 0.9), 30)

                            raw_fx1 = min(raw_fx1, max(0, int(min_kp_x - pad_x)))
                            raw_fy1 = min(raw_fy1, max(0, int(min_kp_y - pad_y)))
                            raw_fx2 = max(raw_fx2, min(w, int(max_kp_x + pad_x)))
                            raw_fy2 = max(raw_fy2, min(h, int(max_kp_y + pad_y)))

                    raw_roi = (float(raw_fx1), float(raw_fy1), float(raw_fx2), float(raw_fy2))

                    # Suavizado exponencial (EMA) continuo cuadro a cuadro
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

        # 3. Aplicar Difuminado Elíptico de Rostros
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
                            small_w = max(1, rw // 10)
                            small_h = max(1, rh // 10)
                            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                            pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

                            k_w = max(15, (rw // 2) * 2 + 1)
                            k_h = max(15, (rh // 2) * 2 + 1)
                            blurred_roi = cv2.GaussianBlur(pix, (k_w, k_h), 0)

                            mask_ellipse = np.zeros((rh, rw), dtype=np.uint8)
                            center = (rw // 2, rh // 2)
                            axes = (int(rw * 0.52), int(rh * 0.52))
                            cv2.ellipse(mask_ellipse, center, axes, 0, 0, 360, 255, -1)
                            mask_ellipse = cv2.GaussianBlur(mask_ellipse, (9, 9), 0)
                            alpha_m = (mask_ellipse.astype(np.float32) / 255.0)[:, :, None]

                            blended_roi = (blurred_roi.astype(np.float32) * alpha_m + roi.astype(np.float32) * (1.0 - alpha_m)).astype(np.uint8)
                            frame[fy1:fy2, fx1:fx2] = blended_roi
                else:
                    tracks_a_eliminar.append(tid)

            for tid in tracks_a_eliminar:
                del estado_rostros[tid]

        # 4. Dibujar Detecciones y Pose Corporal (Sin letreros de Pick/Put)
        if frame_idx in df_por_frame:
            for _, row in df_por_frame[frame_idx].iterrows():
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                track_id = int(row["track_id"])
                conf = float(row.get("conf", 0.0))
                color = color_para_id(track_id)

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id} ({conf:.2f})", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Punto de apoyo pies
                xf, yf = int(row["x_foot"]), int(row["y_foot"])
                cv2.circle(frame, (xf, yf), 5, color, -1)

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

        # 5. Dashboard Superior HUD
        t_act_s = frame_idx / fps
        pct = (frame_idx / max(frames_a_procesar, 1)) * 100

        cv2.rectangle(frame, (0, 0), (w, 38), (15, 15, 15), -1)
        hud_txt = f"ANALISIS RETAIL 15S (YUNET) | Tiempo: {format_mmss(t_act_s)} / {format_mmss(duracion_proc_s)} (FPS: {fps:.1f})"
        cv2.putText(frame, hud_txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 25 == 0 or frame_idx == frames_a_procesar:
            print(f"  Procesando cuadro {frame_idx}/{frames_a_procesar} ({pct:.1f}%)...", end="\r")

    cap.release()
    out.release()

    print("\n" + "=" * 65)
    print(f"[OK] Video anotado de 15 segundos guardado exitosamente:")
    print(f"     Path: {os.path.abspath(salida_path)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anotador de Video Retail con YuNet (Sin Frame Skip, 15s)")
    parser.add_argument("--video", required=True, help="Ruta del video original")
    parser.add_argument("--parquet", default=None, help="Ruta del archivo .parquet de tracking (opcional)")
    parser.add_argument("--zonas", default=None, help="Ruta del archivo .json de zonas")
    parser.add_argument("--carpeta", default="videos_anotados", help="Carpeta de destino para el video")
    parser.add_argument("--salida", default=None, help="Nombre personalizado del archivo de salida")
    parser.add_argument("--max-segundos", type=float, default=15.0, help="Duración máxima en segundos a procesar (default 15.0)")
    parser.add_argument("--no-blur", action="store_true", help="Desactivar difuminado de rostros")
    parser.add_argument("--yunet-model", default="face_detection_yunet_2023mar.onnx", help="Ruta al modelo YuNet .onnx")
    args = parser.parse_args()

    anotar_video_yunet_15s(
        video_path=args.video,
        parquet_path=args.parquet,
        zonas_path=args.zonas,
        carpeta_salida=args.carpeta,
        salida_filename=args.salida,
        max_segundos=args.max_segundos,
        blur_faces=not args.no_blur,
        yunet_model_path=args.yunet_model
    )
