"""
Pipeline unificado de extracción de datos anonimizados + Modo Live en tiempo real.

Modos:
  --modo live    -> Visualización interactiva en tiempo real con cajas, IDs,
                    punto de pies y esqueleto de keypoints corporales (sin rostro).
  --modo guardar -> Inferencia ultra rápida a disco (Parquet) optimizada con cap.grab(),
                    PyTorch inference_mode() y GPU CUDA.

Privacidad por diseño:
- No se guardan frames ni recortes.
- No se persisten keypoints de rostro (0-4), solo los 12 puntos corporales.
"""

import os
import cv2
import time
import json
import logging
import argparse
import colorsys
import warnings
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Silenciar advertencias de Python y el Logger interno de Ultralytics
warnings.filterwarnings("ignore")
LOGGER.setLevel(logging.ERROR)

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
    (5, 6),            # Hombro izq - Hombro der
    (5, 7), (7, 9),    # Brazo izq
    (6, 8), (8, 10),   # Brazo der
    (5, 11), (6, 12),  # Torso
    (11, 12),          # Cadera izq - Cadera der
    (11, 13), (13, 15),# Pierna izq
    (12, 14), (14, 16),# Pierna der
]

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def format_mmss(segundos):
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"


def normalizar_iluminacion(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def color_para_id(track_id):
    if track_id < 0:
        return (200, 200, 200)
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.9)
    return int(b * 255), int(g * 255), int(r * 255)


def dibujar_pose_y_caja(frame, box, track_id, conf, keypoints_frame, kp_confs_frame=None):
    x1, y1, x2, y2 = map(int, box)
    color = color_para_id(track_id)

    # 1. Caja delimitadora
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    etiqueta = f"ID {track_id}" if track_id >= 0 else "Persona"
    cv2.putText(frame, f"{etiqueta} ({conf:.2f})", (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 2. Punto de apoyo en el suelo (pies)
    x_foot, y_foot = int((x1 + x2) / 2), y2
    cv2.circle(frame, (x_foot, y_foot), 6, color, -1)
    cv2.circle(frame, (x_foot, y_foot), 6, (255, 255, 255), 1)

    # 3. Puntos de esqueleto corporal (Keypoints)
    if keypoints_frame is not None and len(keypoints_frame) >= 17:
        for idx1, idx2 in ESQUELETO_CUERPO:
            pt1, pt2 = keypoints_frame[idx1], keypoints_frame[idx2]
            conf1 = kp_confs_frame[idx1] if kp_confs_frame is not None else 1.0
            conf2 = kp_confs_frame[idx2] if kp_confs_frame is not None else 1.0
            if conf1 > 0.3 and conf2 > 0.3 and pt1[0] > 0 and pt2[0] > 0:
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)

        for idx_kp in KEYPOINTS_CUERPO.keys():
            kx, ky = keypoints_frame[idx_kp]
            kp_c = kp_confs_frame[idx_kp] if kp_confs_frame is not None else 1.0
            if kp_c > 0.3 and kx > 0 and ky > 0:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 255), -1)


def modo_live(video_path, modelo_path="yolo26n-pose.pt", conf=0.20, imgsz=640,
              aplicar_clahe=False, tracker="bytetrack_largo.yaml"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    print(f"Cargando modelo YOLO: {modelo_path} en dispositivo {dispositivo}...")
    model = YOLO(modelo_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video")

    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_total_s = total_frames / fps if total_frames > 0 else 1.0

    nombre_win = "Tracking en Vivo - Puntos de Cuerpo (Presiona 'q' para salir)"
    cv2.namedWindow(nombre_win, cv2.WINDOW_NORMAL)
    scale = min(1280 / max(w_orig, 1), 720 / max(h_orig, 1), 1.0)
    cv2.resizeWindow(nombre_win, int(w_orig * scale), int(h_orig * scale))

    frame_idx = 0
    t_inicio = time.time()

    print(f"\n🎥 Modo Live iniciado: {total_frames} frames ({format_mmss(duracion_total_s)} duracion total a {fps:.1f} FPS)")
    print("   Presiona 'q' en la ventana de video para salir.\n")

    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_proc = normalizar_iluminacion(frame) if aplicar_clahe else frame

            results = model.track(
                frame_proc, tracker=tracker, persist=True,
                classes=[0], conf=conf, imgsz=imgsz, verbose=False,
                device=dispositivo
            )
            r = results[0]

            if r.boxes.id is not None and r.keypoints is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.int().cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                keypoints = r.keypoints.xy.cpu().numpy()
                kp_confs = (r.keypoints.conf.cpu().numpy()
                            if r.keypoints.conf is not None else None)

                for i, (box, track_id, confidence) in enumerate(zip(boxes, ids, confs)):
                    kp_i = keypoints[i]
                    kp_c_i = kp_confs[i] if kp_confs is not None else None
                    dibujar_pose_y_caja(frame, box, int(track_id), confidence, kp_i, kp_c_i)

            t_elapsed = time.time() - t_inicio
            fps_live = (frame_idx + 1) / t_elapsed if t_elapsed > 0 else 0
            t_actual_s = frame_idx / fps
            pct = (frame_idx / max(total_frames, 1)) * 100

            # Barra superior de progreso
            cv2.rectangle(frame, (0, 0), (w_orig, 40), (15, 15, 15), -1)
            info_txt = (f"Frame: {frame_idx}/{total_frames} ({pct:.1f}%) | "
                        f"Tiempo: {format_mmss(t_actual_s)} / {format_mmss(duracion_total_s)} ({t_actual_s:.1f}s/{duracion_total_s:.1f}s) | "
                        f"FPS live: {fps_live:.1f}")
            cv2.putText(frame, info_txt, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

            cv2.imshow(nombre_win, frame)
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Cerrado por el usuario.")
                break

    cap.release()
    cv2.destroyAllWindows()


def modo_guardar(video_path, camera_id, modelo_path="yolo26n-pose.pt",
                 imgsz=640, conf=0.25, salida_parquet=None, carpeta_salida="datos_parquet",
                 aplicar_clahe=False, frame_skip=2, tracker="bytetrack_largo.yaml"):

    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No se detectó GPU CUDA, usando CPU.")

    model = YOLO(modelo_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_total_s = total_frames / fps if total_frames > 0 else 1.0

    print(f"Video: {total_frames} frames a {fps:.1f} fps (Duración: {format_mmss(duracion_total_s)} | {duracion_total_s:.1f}s)")
    print(f"Procesando 1 de cada {frame_skip} frames (~{total_frames // frame_skip} a analizar)...")
    print("-" * 60)

    filas = []
    frame_idx = 0
    frames_procesados = 0
    t_inicio = time.time()

    with torch.inference_mode():
        while frame_idx < total_frames:
            if frame_idx % frame_skip != 0:
                ret = cap.grab()
                if not ret:
                    break
                frame_idx += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            frame_proc = normalizar_iluminacion(frame) if aplicar_clahe else frame
            results = model.track(
                frame_proc, tracker=tracker, persist=True,
                classes=[0], conf=conf, imgsz=imgsz, verbose=False,
                device=dispositivo
            )
            r = results[0]

            if r.boxes.id is not None and r.keypoints is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.int().cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                keypoints = r.keypoints.xy.cpu().numpy()
                kp_confs = (r.keypoints.conf.cpu().numpy()
                            if r.keypoints.conf is not None else None)

                for i, (box, track_id, confidence) in enumerate(zip(boxes, ids, confs)):
                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2.0
                    y_center = (y1 + y2) / 2.0
                    x_foot = (x1 + x2) / 2.0
                    y_foot = y2

                    fila = {
                        "camera_id": camera_id,
                        "frame_idx": frame_idx,
                        "timestamp_s": frame_idx / fps,
                        "track_id": int(track_id),
                        "x_foot": float(x_foot), "y_foot": float(y_foot),
                        "x_center": float(x_center), "y_center": float(y_center),
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                        "conf": float(confidence),
                    }

                    for idx_kp, nombre in KEYPOINTS_CUERPO.items():
                        kx, ky = keypoints[i][idx_kp]
                        fila[f"{nombre}_x"] = float(kx)
                        fila[f"{nombre}_y"] = float(ky)
                        if kp_confs is not None:
                            fila[f"{nombre}_conf"] = float(kp_confs[i][idx_kp])

                    filas.append(fila)

            frames_procesados += 1
            frame_idx += 1

            if frames_procesados % 30 == 0:
                elapsed = time.time() - t_inicio
                fps_proc = frames_procesados / elapsed
                pct = (frame_idx / max(total_frames, 1)) * 100
                t_act_s = frame_idx / fps
                restante_s = (total_frames - frame_idx) / (fps_proc * frame_skip) if fps_proc > 0 else 0
                print(f"Frame {frame_idx}/{total_frames} ({pct:.1f}%) | Tiempo video: {format_mmss(t_act_s)}/{format_mmss(duracion_total_s)} | "
                      f"{fps_proc:.1f} fps proc | Transcurrido: {elapsed:.1f}s | ETA: {restante_s:.1f}s", end="\r")

    cap.release()
    df = pd.DataFrame(filas)

    if salida_parquet is None:
        salida_parquet = os.path.join(carpeta_salida, f"tracking_{camera_id}.parquet")

    dir_padre = os.path.dirname(salida_parquet)
    if dir_padre:
        os.makedirs(dir_padre, exist_ok=True)

    df.to_parquet(salida_parquet, index=False)
    t_total = time.time() - t_inicio

    print("\n")
    print("=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"Guardado exitosamente en: {salida_parquet}")
    print(f"Frames totales del video: {total_frames} (Duración: {format_mmss(duracion_total_s)})")
    print(f"Frames analizados: {frames_procesados}")
    print(f"Detecciones totales: {len(df)}")
    print(f"Track IDs únicos: {df['track_id'].nunique() if len(df) else 0}")
    print(f"Tiempo total de procesamiento: {t_total:.2f}s ({t_total/60:.2f} min)")
    print(f"Velocidad promedio: {frames_procesados/t_total:.1f} fps proc")
    print("=" * 60)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", choices=["live", "guardar"], default="guardar",
                        help="'live' = ventana en vivo sin guardar. 'guardar' = procesar a Parquet.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera_id", default="cam_01")
    parser.add_argument("--modelo", default="yolo26n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza (default: 0.25 para no perder tracking en cruces)")
    parser.add_argument("--salida", default=None, help="Ruta directa del .parquet (opcional)")
    parser.add_argument("--carpeta", default="datos_parquet", help="Carpeta de destino para guardar el parquet")
    parser.add_argument("--clahe", action="store_true", help="activa normalización CLAHE")
    parser.add_argument("--frame_skip", type=int, default=2, help="procesa 1 de cada N frames (default: 2)")
    parser.add_argument("--tracker", default="bytetrack_largo.yaml")
    args = parser.parse_args()

    if args.modo == "live":
        modo_live(args.video, modelo_path=args.modelo, conf=args.conf, imgsz=args.imgsz,
                  aplicar_clahe=args.clahe, tracker=args.tracker)
    else:
        modo_guardar(args.video, args.camera_id, modelo_path=args.modelo, imgsz=args.imgsz,
                     conf=args.conf, salida_parquet=args.salida, carpeta_salida=args.carpeta,
                     aplicar_clahe=args.clahe, frame_skip=args.frame_skip, tracker=args.tracker)