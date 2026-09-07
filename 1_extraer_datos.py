"""
Unified pipeline for anonymized pose-tracking data extraction + real-time Live mode.

This module powers the "Góndola Inteligente" computer vision system: a multi-camera
pipeline that tracks customer body movement in a retail environment (e.g. a
supermarket) to detect pick-up / put-back interactions with shelf products,
without ever persisting any personally identifiable visual data.

Modes
-----
--modo live     -> Interactive real-time visualization with bounding boxes, track
                   IDs, a foot-contact point, and the body skeleton (face excluded).
                   Intended for debugging / demoing the tracker, not for production
                   data collection (nothing is written to disk).
--modo guardar  -> High-throughput inference to disk (Parquet), optimized with
                   cap.grab() frame skipping, PyTorch inference_mode(), and CUDA
                   GPU acceleration. This is the production data-collection path.

Privacy by design (Colombian Law 1581 of 2012 — personal data protection)
---------------------------------------------------------------------
- Raw video frames and image crops are never written to disk; only numeric
  pose coordinates are persisted.
- Face-related keypoints (COCO indices 0-4: nose, left_eye, right_eye,
  left_ear, right_ear) are excluded by design. Only the 12 body keypoints
  (indices 5-16: shoulders, elbows, wrists, hips, knees, ankles) are ever
  written to the output Parquet file. See KEYPOINTS_CUERPO below — the face
  indices are structurally absent from that dictionary, so no code path can
  accidentally persist them.
"""

import os
import cv2
import time
import sys
import logging
import argparse
import colorsys
import warnings
import threading
import queue
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Silence noisy third-party warnings and Ultralytics' internal logger so that
# our own progress prints (below) stay readable in the console.
warnings.filterwarnings("ignore")
LOGGER.setLevel(logging.ERROR)

# --- Logging configuration --------------------------------------------------
logger = logging.getLogger("RetailTracker.ExtraerDatos")

def configurar_logger(log_file="pipeline_extraer_datos.log", log_level=logging.INFO):
    """
    Configura el sistema de logging para consola y archivo de registros.
    Garantiza que cualquier fallo o advertencia en cualquier punto del pipeline
    quede registrado con timestamp, nivel de severidad y el detalle del stack trace.
    """
    logger.setLevel(log_level)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"⚠️ No se pudo inicializar el archivo de log '{log_file}': {e}", file=sys.stderr)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)



class ThreadedFrameReader:
    """
    Asynchronous, multi-threaded video frame reader.

    Runs video decoding on a background thread so that disk I/O and OpenCV's
    frame decoding never block the main inference loop. This decouples "how
    fast we can read frames from disk" from "how fast the GPU can run
    inference on them", which is what lets modo_guardar sustain high
    effective throughput on long recordings.

    Frame skipping (frame_skip > 1) is implemented with cv2.VideoCapture.grab(),
    which advances the video position WITHOUT decoding the full BGR image
    matrix for the skipped frames. This is significantly cheaper than reading
    and discarding every frame, since decoding is the expensive part of the
    read() call.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    frame_skip : int
        Process 1 out of every `frame_skip` frames. A value of 1 disables
        skipping (every frame is decoded and queued).
    queue_size : int
        Maximum number of decoded frames buffered ahead of the consumer.
        Bounds memory usage while still allowing the reader thread to stay
        ahead of a (possibly slower) GPU inference loop.
    """

    def __init__(self, video_path, frame_skip=1, queue_size=32):
        self.video_path = video_path
        self.frame_skip = max(frame_skip, 1)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"ThreadedFrameReader: No se pudo abrir el archivo de video en {video_path}")
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)

    def start(self):
        """Start the background decoding thread and return self (fluent API)."""
        self.thread.start()
        return self

    def _reader_loop(self):
        """
        Background thread body: continuously decode frames and push them onto
        the queue, skipping (frame_skip - 1) frames after each decoded frame
        using the cheap .grab() call. Pushes a (None, -1) sentinel when the
        video ends, so the consumer knows to stop.

        Note: `frame_idx` here tracks the RAW frame index of the original
        video (it increments for skipped frames too), not the count of
        frames actually decoded and processed. Downstream code relies on
        this raw index to convert frame gaps back into real elapsed seconds.
        """
        frame_idx = 0
        try:
            while not self.stopped:
                ret, frame = self.cap.read()
                if not ret:
                    self.queue.put((None, -1))
                    break
                self.queue.put((frame, frame_idx))
                frame_idx += 1

                # Fast-forward past skipped frames without decoding their pixel data.
                if self.frame_skip > 1:
                    for _ in range(self.frame_skip - 1):
                        if not self.cap.grab():
                            break
                        frame_idx += 1
        except Exception as e:
            logger.error(f"Error en el hilo decodificador de frames (ThreadedFrameReader) en frame {frame_idx}: {e}", exc_info=True)
            self.queue.put((None, -1))
        finally:
            if self.cap.isOpened():
                self.cap.release()

    def read(self):
        """Block until the next (frame, frame_idx) tuple is available and return it."""
        return self.queue.get()

    def stop(self):
        """Signal the reader thread to stop and release the underlying capture."""
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()


# --- Privacy by design (Colombian Law 1581 of 2012) -------------------------
# Face-related biometric keypoints (COCO indices 0-4: nose, left_eye,
# right_eye, left_ear, right_ear) are intentionally EXCLUDED from this
# dictionary. Only the 12 body keypoints (indices 5-16) are ever iterated
# over when building output rows, so face coordinates can never reach the
# Parquet file — the exclusion is structural, not just a runtime check.
KEYPOINTS_CUERPO = {
    5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle",
}

# Pairs of keypoint indices connected by a line when drawing the body
# skeleton overlay in Live mode. Deliberately covers only torso/limb joints
# (indices 5-16) — no face connections are defined.
ESQUELETO_CUERPO = [
    (5, 6),             # left shoulder - right shoulder
    (5, 7), (7, 9),     # left arm (shoulder -> elbow -> wrist)
    (6, 8), (8, 10),    # right arm
    (5, 11), (6, 12),   # torso (shoulder -> hip, both sides)
    (11, 12),           # left hip - right hip
    (11, 13), (13, 15), # left leg (hip -> knee -> ankle)
    (12, 14), (14, 16), # right leg
]

def format_mmss(segundos):
    """Format a duration in seconds as a zero-padded MM:SS string."""
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"


def color_para_id(track_id):
    """
    Deterministically derive a distinct BGR display color from a track ID.

    Uses the golden ratio to space hues around the color wheel so that
    consecutive IDs get visually distinguishable colors, then converts
    HSV -> RGB -> BGR (OpenCV's channel order).

    Parameters
    ----------
    track_id : int
        Track identifier. A negative ID (no active track) returns a neutral
        gray instead of a hashed color.

    Returns
    -------
    tuple[int, int, int]
        BGR color tuple in the 0-255 range.
    """
    if track_id < 0:
        return (200, 200, 200)
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.9)
    return int(b * 255), int(g * 255), int(r * 255)


def dibujar_pose_y_caja(frame, box, track_id, conf, keypoints_frame, kp_confs_frame=None):
    """
    Draw a single detection's overlay onto `frame` in place: bounding box,
    ID/confidence label, a foot-contact marker, and the body skeleton
    (face keypoints are never drawn — only indices 5-16 are referenced).

    This function is display-only and used exclusively by Live mode; none
    of the pixels it draws are ever persisted to disk.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame to draw on (modified in place).
    box : array-like of 4 floats
        Bounding box in (x1, y1, x2, y2) pixel coordinates.
    track_id : int
        Track ID to display; a negative value is rendered as generic "Persona".
    conf : float
        Detection confidence score, shown alongside the ID.
    keypoints_frame : np.ndarray
        Array of (x, y) keypoint coordinates for this detection (17 x 2,
        COCO layout), though only indices 5-16 are drawn.
    kp_confs_frame : np.ndarray, optional
        Per-keypoint confidence scores, same indexing as keypoints_frame.
        Keypoints below a 0.3 confidence threshold are skipped to avoid
        drawing noisy/unreliable joints.
    """
    x1, y1, x2, y2 = map(int, box)
    color = color_para_id(track_id)

    # 1. Bounding box + label.
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    etiqueta = f"ID {track_id}" if track_id >= 0 else "Persona"
    cv2.putText(frame, f"{etiqueta} ({conf:.2f})", (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 2. Ground-contact point (bottom-center of the box, i.e. the feet),
    #    used elsewhere as the person's floor position for zone/dwell logic.
    x_foot, y_foot = int((x1 + x2) / 2), y2
    cv2.circle(frame, (x_foot, y_foot), 6, color, -1)
    cv2.circle(frame, (x_foot, y_foot), 6, (255, 255, 255), 1)

    # 3. Body skeleton overlay (face keypoints 0-4 are structurally excluded).
    if keypoints_frame is not None and len(keypoints_frame) >= 17:
        for idx1, idx2 in ESQUELETO_CUERPO:
            pt1, pt2 = keypoints_frame[idx1], keypoints_frame[idx2]
            conf1 = kp_confs_frame[idx1] if kp_confs_frame is not None else 1.0
            conf2 = kp_confs_frame[idx2] if kp_confs_frame is not None else 1.0
            if conf1 > 0.3 and conf2 > 0.3 and pt1[0] > 0 and pt2[0] > 0:
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)

        # Body joint markers (indices 5-16: shoulders, elbows, wrists, hips,
        # knees, ankles). Face indices (0-4) are never in range(5, 17).
        for idx_kp in range(5, 17):
            kx, ky = keypoints_frame[idx_kp]
            kp_c = kp_confs_frame[idx_kp] if kp_confs_frame is not None else 1.0
            if kp_c > 0.3 and kx > 0 and ky > 0:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 255), -1)


def modo_live(video_path, modelo_path="yolo26n-pose.pt", conf=0.20, imgsz=640,
              tracker="botsort.yaml"):
    """
    Run the tracker interactively in a live OpenCV window, for debugging and
    demoing. Nothing is written to disk in this mode — it exists purely to
    visually validate detection/tracking quality before running modo_guardar
    on the same footage (both modes share the same tracker config, so what
    you see here reflects the tracking behavior that production runs will
    also use).
    """
    if not os.path.exists(video_path):
        err_msg = f"No se encontró el archivo de video: {video_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    usar_half = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(4)

    logger.info(f"Iniciando modo Live. Cargando modelo YOLO '{modelo_path}' en dispositivo '{dispositivo}' (FP16: {usar_half})...")
    try:
        model = YOLO(modelo_path)
        if hasattr(model, "fuse"):
            try:
                model.fuse()
            except Exception as e:
                logger.warning(f"No se pudo aplicar model.fuse(): {e}")
    except Exception as e:
        logger.error(f"Fallo crítico al cargar el modelo YOLO '{modelo_path}': {e}", exc_info=True)
        raise

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        err_msg = f"No se pudo abrir el archivo de video con OpenCV: {video_path}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    try:
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duracion_total_s = total_frames / fps if total_frames > 0 else 1.0

        nombre_win = "Tracking en Vivo - Puntos de Cuerpo (Presiona 'q' para salir)"
        try:
            cv2.namedWindow(nombre_win, cv2.WINDOW_NORMAL)
            scale = min(1280 / max(w_orig, 1), 720 / max(h_orig, 1), 1.0)
            cv2.resizeWindow(nombre_win, int(w_orig * scale), int(h_orig * scale))
        except Exception as e:
            logger.warning(f"No se pudo ajustar el tamaño de la ventana OpenCV: {e}")

        frame_idx = 0
        t_inicio = time.time()

        print(f"\n🎥 Modo Live iniciado: {total_frames} frames ({format_mmss(duracion_total_s)} duracion total a {fps:.1f} FPS)")
        print("   Presiona 'q' en la ventana de video para salir.\n")

        dir_actual = os.path.dirname(os.path.abspath(__file__))
        tracker_path = tracker
        for candid in [tracker, os.path.join(dir_actual, tracker),
                       os.path.join(dir_actual, "bytetrack_largo.YAML"),
                       os.path.join(dir_actual, "bytetrack_largo.yaml")]:
            if os.path.exists(candid):
                tracker_path = os.path.abspath(candid)
                break
        else:
            logger.warning(f"No se encontró el archivo de tracker '{tracker}'. Ultralytics intentará buscarlo por defecto.")

        with torch.inference_mode():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    results = model.track(
                        frame, tracker=tracker_path, persist=True,
                        classes=[0], conf=conf, imgsz=imgsz, verbose=False,
                        device=dispositivo, half=usar_half
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

                except Exception as e:
                    logger.error(f"Error procesando frame {frame_idx} en modo Live: {e}", exc_info=True)

                t_elapsed = time.time() - t_inicio
                fps_live = (frame_idx + 1) / t_elapsed if t_elapsed > 0 else 0
                t_actual_s = frame_idx / fps
                pct = (frame_idx / max(total_frames, 1)) * 100

                cv2.rectangle(frame, (0, 0), (w_orig, 40), (15, 15, 15), -1)
                info_txt = (f"Frame: {frame_idx}/{total_frames} ({pct:.1f}%) | "
                            f"Tiempo: {format_mmss(t_actual_s)} / {format_mmss(duracion_total_s)} ({t_actual_s:.1f}s/{duracion_total_s:.1f}s) | "
                            f"FPS live: {fps_live:.1f}")
                cv2.putText(frame, info_txt, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

                cv2.imshow(nombre_win, frame)
                frame_idx += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Modo Live interrumpido manualmente por el usuario.")
                    break
    except Exception as e:
        logger.critical(f"Fallo catastrófico en modo_live: {e}", exc_info=True)
        raise
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


class DisjointSet:
    """Estructura de conjuntos disjuntos (Union-Find) optimizada con compresión de caminos."""
    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i


def limpiar_y_renumerar_ids(df, min_duracion_s=1, max_gap_s=2, max_dist_px=80.0,
                             fps_efectivo=25.0, fps_crudo=None):
    """
    Time-based adaptive track consolidation and ID cleanup.
    """
    if df.empty or "track_id" not in df.columns:
        return df

    try:
        fps_efectivo = max(float(fps_efectivo), 1.0)
        fps_crudo = max(float(fps_crudo if fps_crudo is not None else fps_efectivo), 1.0)

        min_frames_track = max(1, int(round(min_duracion_s * fps_efectivo)))
        max_gap_frames = max(1, int(round(max_gap_s * fps_crudo)))

        df = df.copy()

        # 1. Ephemeral noise filtering: drop tracks with too few processed samples.
        conteo_frames = df.groupby("track_id")["frame_idx"].nunique()
        ids_validos = conteo_frames[conteo_frames >= min_frames_track].index
        df = df[df["track_id"].isin(ids_validos)].copy()

        if df.empty:
            return df

        # 2. Trajectory stitching (Optimizada O(N log N) con DSU y salida temprana)
        stats = []
        for tid, grp in df.groupby("track_id"):
            grp = grp.sort_values("frame_idx")
            stats.append({
                "track_id": tid,
                "f_start": grp["frame_idx"].iloc[0],
                "f_end": grp["frame_idx"].iloc[-1],
                "x_start": grp["x_foot"].iloc[0],
                "y_start": grp["y_foot"].iloc[0],
                "x_end": grp["x_foot"].iloc[-1],
                "y_end": grp["y_foot"].iloc[-1],
            })

        df_s = pd.DataFrame(stats).sort_values("f_start").reset_index(drop=True)
        dsu = DisjointSet(df_s["track_id"])

        records = df_s.to_dict("records")
        n_tracks = len(records)

        for i in range(n_tracks):
            rec_i = records[i]
            t_i = rec_i["track_id"]
            f_end_i = rec_i["f_end"]
            x_end_i = rec_i["x_end"]
            y_end_i = rec_i["y_end"]

            for j in range(i + 1, n_tracks):
                rec_j = records[j]
                gap = rec_j["f_start"] - f_end_i

                # Como los registros están ordenados por f_start, si el inicio de j
                # supera la brecha máxima, ningún j posterior podrá cumplir la condición.
                if gap > max_gap_frames:
                    break

                if gap >= 0:
                    t_j = rec_j["track_id"]
                    dist = np.sqrt((rec_j["x_start"] - x_end_i)**2 + (rec_j["y_start"] - y_end_i)**2)
                    if dist <= max_dist_px:
                        dsu.union(t_i, t_j)

        remapping = {tid: dsu.find(tid) for tid in df_s["track_id"]}
        df["track_id"] = df["track_id"].map(remapping)

        # 3. Clean sequential renumbering
        orden_aparicion = df.groupby("track_id")["frame_idx"].min().sort_values().index
        mapeo_limpio = {old_id: idx + 1 for idx, old_id in enumerate(orden_aparicion)}
        df["track_id"] = df["track_id"].map(mapeo_limpio)

        return df
    except Exception as e:
        logger.error(f"Error durante el procesamiento de unificación de IDs (limpiar_y_renumerar_ids): {e}", exc_info=True)
        return df


def modo_guardar(video_path, camera_id, modelo_path="yolo26n-pose.pt",
                 imgsz=640, conf=0.35, salida_parquet=None, carpeta_salida="datos_parquet",
                 frame_skip=2, tracker="botsort.yaml",
                 dispositivo="auto", min_duracion_s=1.0, max_gap_s=4.0, max_dist_px=120.0):
    """
    Run the production pose-tracking pipeline over a video and persist the
    resulting (privacy-filtered) per-detection data to a Parquet file.
    """
    if not os.path.exists(video_path):
        err_msg = f"No se encontró el archivo de video: {video_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"Iniciando modo Guardar para video '{video_path}' (camera_id={camera_id})...")

    try:
        # --- Device selection -----------------------------------------------
        es_openvino = "openvino" in str(modelo_path).lower() or os.path.isdir(modelo_path)
        disp_str = str(dispositivo).lower().strip()

        if es_openvino or disp_str in ["cpu", "false"]:
            dispositivo_real = "cpu"
            usar_half = False
            if hasattr(torch, "set_num_threads"):
                try:
                    torch.set_num_threads(min(8, os.cpu_count() or 4))
                except Exception:
                    pass
            logger.info(f"Inferencia asignada a CPU -> Hilos vectoriales configurados ({torch.get_num_threads()} hilos)")
        elif disp_str in ["cuda", "gpu", "0", "true"]:
            if torch.cuda.is_available():
                dispositivo_real = 0
                usar_half = True
                logger.info(f"GPU NVIDIA forzada ({torch.cuda.get_device_name(0)}) | Precisión FP16")
                torch.backends.cudnn.benchmark = True
            else:
                dispositivo_real = "cpu"
                usar_half = False
                logger.warning("GPU forzada pero CUDA no está disponible. Usando CPU.")
        else:
            if torch.cuda.is_available():
                dispositivo_real = 0
                usar_half = True
                logger.info(f"GPU detectada automáticamente: {torch.cuda.get_device_name(0)} | Precisión FP16")
                torch.backends.cudnn.benchmark = True
            else:
                dispositivo_real = "cpu"
                usar_half = False
                logger.info("Inferencia en CPU (Sin GPU CUDA detectada)")

        cv2.setNumThreads(4)

        try:
            logger.info(f"Cargando modelo YOLO pose desde '{modelo_path}'...")
            model = YOLO(modelo_path)
            if hasattr(model, "fuse"):
                try:
                    model.fuse()
                except Exception as e:
                    logger.warning(f"No se pudo fusionar capas del modelo (model.fuse): {e}")
        except Exception as e:
            logger.error(f"Error crítico cargando el modelo '{modelo_path}': {e}", exc_info=True)
            raise

        # --- Video metadata ----------------------------------------------------
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            err_msg = f"No se pudo abrir el archivo de video con OpenCV: {video_path}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duracion_total_s = total_frames / fps if total_frames > 0 else 1.0

        logger.info(f"Video cargado: {total_frames} frames a {fps:.1f} fps (Duración: {format_mmss(duracion_total_s)} | {duracion_total_s:.1f}s)")
        print(f"Video: {total_frames} frames a {fps:.1f} fps (Duración: {format_mmss(duracion_total_s)} | {duracion_total_s:.1f}s)")
        print(f"Inferencia Multihilo: imgsz={imgsz} | vid_stride={frame_skip} (~{total_frames // max(frame_skip, 1)} frames a analizar)...")
        print("-" * 60)

        dir_actual = os.path.dirname(os.path.abspath(__file__))
        tracker_path = tracker
        candidatos = [
            tracker,
            os.path.join(dir_actual, tracker),
            "botsort.yaml",
            "bytetrack_largo.yaml",
            os.path.join(dir_actual, "bytetrack_largo.yaml")
        ]
        for candid in candidatos:
            if os.path.exists(candid):
                tracker_path = os.path.abspath(candid)
                break
        else:
            logger.info(f"Usando tracker configurado: '{tracker}'")

        # --- Warmup -------------------------------------------------------------
        dummy_h = h_orig if h_orig > 0 else imgsz
        dummy_w = w_orig if w_orig > 0 else imgsz
        frame_dummy = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        try:
            _ = model.track(frame_dummy, tracker=tracker_path, persist=True, classes=[0],
                            conf=conf, imgsz=imgsz, device=dispositivo_real, verbose=False)
            if hasattr(model, "predictor") and hasattr(model.predictor, "trackers") and model.predictor.trackers:
                model.predictor.trackers[0].reset()
        except Exception as e:
            logger.warning(f"Advertencia en etapa de calentamiento (warmup): {e}", exc_info=True)

        filas = []
        frames_procesados = 0

        # --- Main processing loop ------------------------------------------------
        try:
            reader = ThreadedFrameReader(video_path, frame_skip=frame_skip, queue_size=32).start()
            logger.info(f"Hilo secundario de decodificación iniciado (Buffer: 32 frames | Hilos OS: {threading.active_count()})")
            print(f"🧵 Multithreading Activo: Hilo secundario de decodificación iniciado en segundo plano (Buffer: 32 frames | Hilos OS activos: {threading.active_count()})", flush=True)
        except Exception as e:
            logger.error(f"Fallo al iniciar ThreadedFrameReader: {e}", exc_info=True)
            raise

        t_inicio = time.time()

        try:
            with torch.inference_mode():
                while True:
                    item = reader.read()
                    if item is None or item[0] is None:
                        break
                    frame, frame_idx = item

                    frames_procesados += 1
                    t_frame_start = time.time()
                    ids = []
                    r = None

                    try:
                        resultados = model.track(
                            frame,
                            tracker=tracker_path,
                            persist=True,
                            classes=[0],
                            conf=conf,
                            imgsz=imgsz,
                            verbose=False,
                            device=dispositivo_real
                        )
                        if resultados and len(resultados) > 0:
                            r = resultados[0]

                        if r.boxes.id is not None and r.keypoints is not None:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            ids = r.boxes.id.int().cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            keypoints = r.keypoints.xy.cpu().numpy()
                            kp_confs = (r.keypoints.conf.cpu().numpy()
                                        if r.keypoints.conf is not None else None)

                            for j, (box, track_id, confidence) in enumerate(zip(boxes, ids, confs)):
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
                                    kx, ky = keypoints[j][idx_kp]
                                    fila[f"{nombre}_x"] = float(kx)
                                    fila[f"{nombre}_y"] = float(ky)
                                    if kp_confs is not None:
                                        fila[f"{nombre}_conf"] = float(kp_confs[j][idx_kp])

                                filas.append(fila)

                    except Exception as e:
                        logger.error(f"Error procesando frame {frame_idx} (timestamp {frame_idx/fps:.2f}s): {e}", exc_info=True)
                        print(f"\n⚠️ Error procesando frame {frame_idx}: {e}. Continuando...", flush=True)
                        continue

                    t_frame_ms = (time.time() - t_frame_start) * 1000.0
                    elapsed = time.time() - t_inicio
                    fps_proc = frames_procesados / max(elapsed, 0.001)
                    pct = (frame_idx / max(total_frames, 1)) * 100
                    t_act_s = frame_idx / fps
                    restante_s = (total_frames - frame_idx) / (fps_proc * max(frame_skip, 1)) if fps_proc > 0 else 0
                    
                    num_personas = len(ids) if ('r' in locals() and r is not None and r.boxes.id is not None and len(r.boxes.id) > 0) else 0
                    lista_ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
                    str_ids = f" (IDs: {lista_ids})" if lista_ids else ""
                    print(f"[Frame {frame_idx}/{total_frames} ({pct:5.1f}%)] ⏱️ {t_frame_ms:4.0f}ms | {fps_proc:4.1f} fps proc | Video: {format_mmss(t_act_s)}/{format_mmss(duracion_total_s)} | Personas: {num_personas}{str_ids} | ETA: {format_mmss(restante_s)}", flush=True)
        finally:
            reader.stop()

        df = pd.DataFrame(filas)

        # --- Post-processing: noise filtering, track stitching, ID cleanup -----
        if not df.empty:
            n_ids_antes = df["track_id"].nunique()
            max_id_antes = df["track_id"].max()
            fps_efectivo = (fps / max(frame_skip, 1)) if fps > 0 else 25.0
            try:
                df = limpiar_y_renumerar_ids(df, min_duracion_s=min_duracion_s, max_gap_s=max_gap_s, max_dist_px=max_dist_px,
                                             fps_efectivo=fps_efectivo, fps_crudo=fps)
                n_ids_despues = df["track_id"].nunique()
                max_id_despues = df["track_id"].max() if not df.empty else 0
                logger.info(f"Limpieza de trazabilidad: {n_ids_antes} IDs -> {n_ids_despues} IDs limpios secuenciales.")
                print(f"\n✨ Limpieza de Trazabilidad: Se eliminó ruido efímero y se unificaron {n_ids_antes - n_ids_despues} IDs fragmentados.")
                print(f"   IDs originales: {n_ids_antes} (máx ID: {max_id_antes}) -> IDs Limpios Secuenciales: {n_ids_despues} personas (IDs: 1 a {max_id_despues}).")
            except Exception as e:
                logger.error(f"Error en post-procesamiento de IDs: {e}", exc_info=True)

        # --- Output ---------------------------------------------------------
        if salida_parquet is None:
            salida_parquet = os.path.join(carpeta_salida, f"tracking_{camera_id}.parquet")

        try:
            dir_padre = os.path.dirname(salida_parquet)
            if dir_padre:
                os.makedirs(dir_padre, exist_ok=True)

            df.to_parquet(salida_parquet, index=False)
            logger.info(f"Guardado exitosamente en Parquet: {salida_parquet} ({len(df)} filas)")
        except Exception as e:
            logger.error(f"Fallo crítico al escribir el archivo Parquet '{salida_parquet}': {e}", exc_info=True)
            raise

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

    except Exception as e:
        logger.critical(f"Fallo no controlado en modo_guardar para video {video_path}: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anonymized pose-tracking pipeline for retail shelf monitoring "
                    "(Góndola Inteligente). Runs in either an interactive 'live' "
                    "preview mode or a headless 'guardar' (save) mode that writes "
                    "privacy-filtered tracking data to Parquet."
    )
    parser.add_argument("--modo", choices=["live", "guardar"], default="guardar",
                        help="'live' = interactive preview window, nothing saved. "
                             "'guardar' = process the full video and write results to Parquet.")
    parser.add_argument("--video", required=True,
                        help="Path to the input video file.")
    parser.add_argument("--camera_id", default="cam_01",
                        help="Identifier for the source camera; stored in every output row "
                             "and used to name the default output file.")
    parser.add_argument("--modelo", default="yolo26n-pose.pt",
                        help="Path or name of the YOLO pose model weights to load.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (longer side, in pixels).")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25, kept low so tracking "
                             "survives crossings/occlusions).")
    parser.add_argument("--salida", default=None,
                        help="Explicit output .parquet path (optional; overrides --carpeta).")
    parser.add_argument("--carpeta", default="datos_parquet",
                        help="Output directory used to build the default output filename.")
    parser.add_argument("--frame_skip", type=int, default=2,
                        help="Process 1 out of every N raw video frames (default: 2).")
    parser.add_argument("--tracker", default="botsort.yaml",
                        help="Filename of the tracker YAML configuration to use.")
    parser.add_argument("--dispositivo", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Inference device: 'auto', 'cuda' (GPU), or 'cpu'.")
    parser.add_argument("--log_file", default="pipeline_extraer_datos.log",
                        help="Path to output log file (default: pipeline_extraer_datos.log).")
    args = parser.parse_args()

    configurar_logger(log_file=args.log_file)

    try:
        if args.modo == "live":
            modo_live(args.video, modelo_path=args.modelo, conf=args.conf, imgsz=args.imgsz,
                      tracker=args.tracker)
        else:
            modo_guardar(args.video, args.camera_id, modelo_path=args.modelo, imgsz=args.imgsz,
                         conf=args.conf, salida_parquet=args.salida, carpeta_salida=args.carpeta,
                         frame_skip=args.frame_skip, tracker=args.tracker,
                         dispositivo=args.dispositivo)
    except Exception as e:
        logger.critical(f"El pipeline finalizó de forma anómala debido a un error: {e}", exc_info=True)
        print(f"\n❌ [ERROR CRÍTICO DEL PIPELINE] {e}. Revisa el archivo '{args.log_file}' para ver el reporte de error detallado.", file=sys.stderr)
        sys.exit(1)