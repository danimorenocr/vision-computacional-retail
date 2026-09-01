"""
Herramienta flexible para recortar clips de video y opcionalmente reducir los FPS (ej: de 60 FPS a 20 o 25 FPS).

Beneficio de reducir FPS:
Videos grabados a 60 FPS generan 2.4x o 4x más fotogramas innecesarios. Reducirlos a 15, 20 o 25 FPS:
- Acelera el seguimiento de YOLO hasta 3x más rápido.
- Reduce considerablemente el tamaño del archivo de video y el Parquet de salida.

Ejemplos de uso:
  1. Cortar clip y reducir de 60 FPS a 20 FPS (Recomendado):
     python cortar_clip.py --video grabacion3.mp4 --inicio 2:20 --fin 2:45 --target_fps 20 --salida clip_test.mp4

  2. Cortar sin cambiar los FPS originales:
     python cortar_clip.py --video videoplayback.mp4 --inicio 0 --fin 30 --salida clip_test.mp4
"""

import os
import cv2
import argparse


def parsear_tiempo(tiempo):
    """
    Convierte entradas de tiempo a segundos (float).
    Ejemplos:
      "30" -> 30.0
      "1:30" -> 90.0
      "02:15:30" -> 8130.0
    """
    if isinstance(tiempo, (int, float)):
        return float(tiempo)
    tiempo_str = str(tiempo).strip()
    if ":" in tiempo_str:
        partes = [float(p) for p in tiempo_str.split(":")]
        if len(partes) == 2:
            return partes[0] * 60 + partes[1]
        elif len(partes) == 3:
            return partes[0] * 3600 + partes[1] * 60 + partes[2]
    return float(tiempo_str)


def cortar_clip(video_path, inicio_str, fin_str=None, duracion_s=None, target_fps=None, salida_path="clip_test.mp4"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps_orig = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_total_video_s = total_frames / fps_orig

    inicio_s = parsear_tiempo(inicio_str)

    if fin_str is not None:
        fin_s = parsear_tiempo(fin_str)
    elif duracion_s is not None:
        fin_s = inicio_s + float(duracion_s)
    else:
        fin_s = inicio_s + 300.0  # 300 segundos por defecto

    inicio_s = max(0.0, min(inicio_s, duracion_total_video_s))
    fin_s = max(inicio_s, min(fin_s, duracion_total_video_s))

    frame_inicio = int(inicio_s * fps_orig)
    frame_fin = int(fin_s * fps_orig)

    # Determinar si se aplica reducción de FPS
    if target_fps is not None and target_fps > 0 and target_fps < fps_orig:
        fps_salida = float(target_fps)
        step_frames = int(round(fps_orig / fps_salida))
    else:
        fps_salida = fps_orig
        step_frames = 1

    # Posicionar el puntero del video directamente en el frame de inicio
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

    dir_salida = os.path.dirname(salida_path)
    if dir_salida:
        os.makedirs(dir_salida, exist_ok=True)

    out = cv2.VideoWriter(salida_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_salida, (w, h))

    print("\n" + "=" * 60)
    print("✂️ HERRAMIENTA DE RECORTE DE VIDEO Y REDUCCIÓN DE FPS")
    print("=" * 60)
    print(f"Video origen: {video_path} ({fps_orig:.1f} FPS orig | {duracion_total_video_s:.1f}s totales | {total_frames} frames)")
    print(f"Rango de corte: {inicio_s:.1f}s ➔ {fin_s:.1f}s (Duración clip: {fin_s - inicio_s:.1f}s)")
    print(f"FPS de salida: {fps_salida:.1f} FPS" + (f" (Reducción activa: 1 de cada {step_frames} frames)" if step_frames > 1 else ""))

    frame_idx = frame_inicio
    frames_escritos = 0

    while frame_idx < frame_fin:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - frame_inicio) % step_frames == 0:
            out.write(frame)
            frames_escritos += 1

        frame_idx += 1

    cap.release()
    out.release()

    print("-" * 60)
    print(f"✅ Clip guardado exitosamente en: {os.path.abspath(salida_path)}")
    print(f"   Frames exportados: {frames_escritos} frames a {fps_salida:.1f} FPS")
    print(f"   Duración real del clip: {frames_escritos / fps_salida:.2f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recorta un fragmento de video y opcionalmente reduce sus FPS.")
    parser.add_argument("--video", default="videoplayback.mp4", help="Ruta del video original")
    parser.add_argument("--inicio", default="0", help="Tiempo de inicio en segundos (ej: 0, 30) o formato min:seg (ej: 1:30)")
    parser.add_argument("--fin", default=None, help="Tiempo de fin en segundos (ej: 90) o formato min:seg (ej: 3:00)")
    parser.add_argument("--duracion", type=float, default=None, help="Duración del clip en segundos a partir del tiempo de inicio")
    parser.add_argument("--target_fps", type=float, default=20.0, help="FPS objetivo de salida (ej: 20 o 25 para acelerar el tracking)")
    parser.add_argument("--salida", default="clip_test.mp4", help="Nombre del archivo de video resultante")
    args = parser.parse_args()

    cortar_clip(args.video, args.inicio, fin_str=args.fin, duracion_s=args.duracion,
                target_fps=args.target_fps, salida_path=args.salida)