"""
Herramienta interactiva para definir zonas (góndolas, pasillos y zonas de exclusión)
haciendo clic sobre el primer frame del video. Guarda las zonas en un archivo zonas_cam_XX.json.

Controles:
- Clic izquierdo: agregar punto al polígono actual
- 'n': cancelar el polígono en construcción
- 'f': marcar polígono como Zona de FLUJO / PASILLO (Verde)
- 'i': marcar polígono como Zona de INTERACCIÓN / GÓNDOLA (Naranja)
- 'e': marcar polígono como Zona de EXCLUSIÓN / IGNORAR (Rojo - Ignora detecciones en esa área)
- 's': guardar zonas y salir
- 'q': salir sin guardar
"""

import cv2
import json
import argparse
import numpy as np


def definir_zonas(video_path, camera_id, salida_json=None):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("No se pudo leer el video")

    frame_original = frame.copy()
    zonas = []
    poligono_actual = []
    contador = {"flujo": 0, "interaccion": 0, "exclusion": 0}

    def click_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            poligono_actual.append([x, y])

    h_orig, w_orig = frame_original.shape[:2]
    max_w, max_h = 1280, 720
    scale = min(max_w / max(w_orig, 1), max_h / max(h_orig, 1), 1.0)
    win_w, win_h = int(w_orig * scale), int(h_orig * scale)

    cv2.namedWindow("Definir zonas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Definir zonas", win_w, win_h)
    cv2.setMouseCallback("Definir zonas", click_callback)

    print("\n" + "=" * 60)
    print("📍 EDITOR INTERACTIVO DE ZONAS Y EXCLUSIONES")
    print("=" * 60)
    print("Controles:")
    print("  - Clic Izquierdo: Marcar punto del polígono")
    print("  - 'f' : Guardar como Zona de FLUJO / PASILLO (Verde)")
    print("  - 'i' : Guardar como Zona de INTERACCIÓN / GÓNDOLA (Naranja)")
    print("  - 'e' : Guardar como Zona de EXCLUSIÓN / IGNORAR (Rojo - Ignora esa área)")
    print("  - 'n' : Cancelar polígono en construcción")
    print("  - 's' : Guardar todo y salir")
    print("  - 'q' : Salir sin guardar\n")

    while True:
        vis = frame_original.copy()

        # Dibujar zonas ya guardadas
        for z in zonas:
            t = z.get("type", "flujo")
            if t == "flujo":
                color = (0, 255, 0)       # Verde
            elif t == "interaccion":
                color = (0, 165, 255)     # Naranja
            elif t == "exclusion":
                color = (0, 0, 255)       # Rojo (Exclusión)
            else:
                color = (255, 255, 0)

            pts = np.array(z["polygon"], dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

            # Relleno semi-transparente para zonas de exclusión
            if t == "exclusion":
                sub = vis.copy()
                cv2.fillPoly(sub, [pts], (0, 0, 180))
                vis = cv2.addWeighted(sub, 0.35, vis, 0.65, 0)

            cv2.putText(vis, z["id"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        # Dibujar polígono en construcción
        for i, p in enumerate(poligono_actual):
            cv2.circle(vis, tuple(p), 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(vis, tuple(poligono_actual[i - 1]), tuple(p), (0, 0, 255), 1)

        cv2.imshow("Definir zonas", vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('f') and len(poligono_actual) >= 3:
            contador["flujo"] += 1
            zona_id = f"flujo_{contador['flujo']}"
            zonas.append({"id": zona_id, "type": "flujo", "polygon": poligono_actual.copy()})
            print(f"✅ Zona guardada [PASILLO/FLUJO]: {zona_id}")
            poligono_actual = []

        elif key == ord('i') and len(poligono_actual) >= 3:
            contador["interaccion"] += 1
            zona_id = f"gondola_{contador['interaccion']}"
            zonas.append({"id": zona_id, "type": "interaccion", "polygon": poligono_actual.copy()})
            print(f"✅ Zona guardada [INTERACCIÓN/GÓNDOLA]: {zona_id}")
            poligono_actual = []

        elif key == ord('e') and len(poligono_actual) >= 3:
            contador["exclusion"] += 1
            zona_id = f"exclusion_{contador['exclusion']}"
            zonas.append({"id": zona_id, "type": "exclusion", "polygon": poligono_actual.copy()})
            print(f"🚫 Zona guardada [EXCLUSIÓN/IGNORAR]: {zona_id}")
            poligono_actual = []

        elif key == ord('n'):
            poligono_actual = []
            print("❌ Polígono en construcción cancelado.")

        elif key == ord('s'):
            break

        elif key == ord('q'):
            zonas = []
            break

    cv2.destroyAllWindows()

    if salida_json is None:
        salida_json = f"zonas_{camera_id}.json"

    with open(salida_json, "w", encoding="utf-8") as f:
        json.dump({"camera_id": camera_id, "zones": zonas}, f, indent=2)

    print(f"\n✅ Guardado exitoso: {salida_json} ({len(zonas)} zonas configuradas)")
    return zonas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera_id", required=True)
    parser.add_argument("--salida", default=None)
    args = parser.parse_args()

    definir_zonas(args.video, args.camera_id, args.salida)