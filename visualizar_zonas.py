"""
Dos visualizaciones para confirmar que las zonas y las métricas tienen sentido:

1. mapa_zonas.png: imagen estática con los polígonos coloreados según qué tan
   "caliente" es la zona (más tiempo total de permanencia = más rojo), con las
   métricas de cada zona escritas encima.

2. video_zonas.mp4: el video original con las zonas dibujadas siempre visibles,
   cada persona con su track_id y un texto de en qué zona está parada, y el
   borde de la góndola resaltado en rojo brillante durante los frames donde
   hay una interacción activa (mano detectada dentro).
"""

import cv2
import numpy as np
import pandas as pd
import json
import argparse


def punto_en_zona(x, y, poligono):
    return cv2.pointPolygonTest(np.array(poligono, dtype=np.int32), (float(x), float(y)), False) >= 0


def mapa_zonas_estatico(video_referencia, zonas_json_path, ranking_csv_path,
                         salida_png="mapa_zonas.png"):
    with open(zonas_json_path) as f:
        zonas_config = json.load(f)

    ranking = {}
    try:
        df_rank = pd.read_csv(ranking_csv_path, index_col=0)
        ranking = df_rank.to_dict(orient="index")
    except FileNotFoundError:
        print("Aviso: no se encontró ranking_zonas.csv, se dibuja sin métricas de calor")

    cap = cv2.VideoCapture(video_referencia)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("No se pudo leer el video de referencia")

    overlay = frame.copy()

    max_tiempo = max([v.get("tiempo_total_s", 0) for v in ranking.values()], default=1) or 1

    for z in zonas_config["zones"]:
        pts = np.array(z["polygon"], dtype=np.int32)

        if z["type"] == "flujo" and z["id"] in ranking:
            intensidad = ranking[z["id"]].get("tiempo_total_s", 0) / max_tiempo
            color = (int(50 * (1 - intensidad)), int(80 * (1 - intensidad)), int(180 + 75 * intensidad))
        elif z["type"] == "interaccion":
            color = (0, 140, 255)
        else:
            color = (0, 200, 0)

        cv2.fillPoly(overlay, [pts], color)

    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    for z in zonas_config["zones"]:
        pts = np.array(z["polygon"], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)

        etiqueta = z["id"]
        if z["id"] in ranking:
            r = ranking[z["id"]]
            etiqueta += f" | {r.get('visitantes_unicos', '?')} vis | {r.get('tiempo_total_s', 0):.1f}s"

        x_texto, y_texto = pts[0]
        cv2.putText(frame, etiqueta, (int(x_texto), int(y_texto) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imwrite(salida_png, frame)
    print(f"Mapa guardado en: {salida_png}")


def video_con_zonas(video_path, parquet_path, zonas_json_path,
                     interacciones_parquet=None, salida_video="video_zonas.mp4"):
    df = pd.read_parquet(parquet_path)
    with open(zonas_json_path) as f:
        zonas_config = json.load(f)

    zonas_flujo = [z for z in zonas_config["zones"] if z["type"] == "flujo"]
    zonas_interaccion = [z for z in zonas_config["zones"] if z["type"] == "interaccion"]

    interacciones_df = None
    if interacciones_parquet:
        try:
            interacciones_df = pd.read_parquet(interacciones_parquet)
        except FileNotFoundError:
            print("Aviso: no se encontró el parquet de interacciones, se omite el resaltado")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(salida_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    df_por_frame = {k: v for k, v in df.groupby("frame_idx")}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        # zonas de interacción activas en este instante (para resaltar)
        zonas_activas_ahora = set()
        if interacciones_df is not None and len(interacciones_df) > 0:
            activas = interacciones_df[(interacciones_df["inicio_s"] <= t) & (interacciones_df["fin_s"] >= t)]
            zonas_activas_ahora = set(activas["zona"].tolist())

        # dibujar zonas
        for z in zonas_flujo:
            pts = np.array(z["polygon"], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 200, 0), 1)

        for z in zonas_interaccion:
            pts = np.array(z["polygon"], dtype=np.int32)
            if z["id"] in zonas_activas_ahora:
                cv2.polylines(frame, [pts], True, (0, 0, 255), 4)  # resaltado ROJO = interacción activa
                cv2.putText(frame, f"INTERACCION: {z['id']}", (pts[0][0], pts[0][1] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.polylines(frame, [pts], True, (0, 140, 255), 2)

        # dibujar personas + zona en la que están paradas
        if frame_idx in df_por_frame:
            for _, row in df_por_frame[frame_idx].iterrows():
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                track_id = int(row["track_id"])

                zona_actual = None
                for z in zonas_flujo:
                    if punto_en_zona(row["x_foot"], row["y_foot"], z["polygon"]):
                        zona_actual = z["id"]
                        break

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                etiqueta = f"id:{track_id}" + (f" [{zona_actual}]" if zona_actual else "")
                cv2.putText(frame, etiqueta, (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"t={t:.2f}s", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Video guardado en: {salida_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--zonas", required=True)
    parser.add_argument("--ranking", default="ranking_zonas.csv")
    parser.add_argument("--interacciones", default="interacciones_gondola.parquet")
    args = parser.parse_args()

    mapa_zonas_estatico(args.video, args.zonas, args.ranking)
    video_con_zonas(args.video, args.parquet, args.zonas, args.interacciones)