"""
===============================================================================
🌐 FUSIÓN MULTICÁMARA Y HOMOGRAFÍA DE RETAIL TRACKER - PLANO 2D UNIFICADO (TOP-DOWN)
===============================================================================
Módulo para calibración espacial de múltiples cámaras:
1. Calibración manual guiada punto a punto (P1..P4).
2. CALIBRACIÓN AUTOMÁTICA POR POLÍGONOS Y IDs DE ZONAS (GÓNDOLAS Y PASILLOS):
   Empareja los vértices de polígonos dibujados con el mismo ID ('gondola_1', 'gondola_2', 'flujo_1')
   en ambas cámaras para calcular la matriz de homografía sin clics manuales.
3. PROYECCIÓN EN MAPA DE PLANTA 2D UNIFICADO (TOP-DOWN BIRDS-EYE VIEW):
   Evita la deformación por perspectiva al mapear todas las cámaras a un plano neutro 2D del piso.
4. DIAGNÓSTICO VISUAL:
   - Heatmap Top-Down 2D del local.
   - Trayectorias 2D en plano cenital.
   - Superposición Óptica 2D Warped (Mezcla 50/50 de perspectivas).
===============================================================================
"""

import os
import cv2
import json
import numpy as np
import pandas as pd

# Paleta de colores para identificadores visuales de puntos de calibración (BGR)
PALETA_BGR = [
    (221, 119, 127), # P1: Morado
    (117, 158, 29),  # P2: Verde
    (48, 90, 216),   # P3: Naranja
    (39, 159, 239),  # P4: Amarillo
    (126, 83, 212),  # P5: Rosado
    (221, 138, 55),  # P6: Azul
]

NOMBRES_PUNTOS = ["P1 (Morado)", "P2 (Verde)", "P3 (Naranja)", "P4 (Amarillo)", "P5 (Rosado)", "P6 (Azul)"]

# Colores distintivos para trayectorias de cada cámara (BGR)
COLORES_CAMARAS = [
    (255, 200, 0),   # Cyan brillante (Cámara Ref)
    (0, 255, 120),   # Verde lima brillante (Cámara Sec 1)
    (255, 80, 180),  # Magenta (Cámara Sec 2)
    (0, 140, 255),   # Naranja (Cámara Sec 3)
]


def dibujar_puntos_color(frame_bgr, puntos, resaltar_ultimo=False):
    """
    Dibuja los puntos de calibración sobre una imagen con colores y etiquetas P1, P2...
    """
    vis = frame_bgr.copy()
    for i, (x, y) in enumerate(puntos):
        color = PALETA_BGR[i % len(PALETA_BGR)]
        es_ultimo = resaltar_ultimo and (i == len(puntos) - 1)
        radio = 12 if es_ultimo else 8

        cv2.circle(vis, (int(x), int(y)), radio, color, -1)
        cv2.circle(vis, (int(x), int(y)), radio + 2, (255, 255, 255), 2)
        if es_ultimo:
            cv2.circle(vis, (int(x), int(y)), radio + 6, color, 2)

        cv2.putText(vis, f"P{i+1}", (int(x) + 14, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return vis


def calcular_homografia(puntos_src, puntos_dst):
    """
    Calcula la matriz de Homografía 3x3 para transformar puntos de la cámara secundaria (src)
    al plano de coordenadas de la cámara de referencia o plano 2D (dst).
    """
    if len(puntos_src) < 4 or len(puntos_dst) < 4:
        raise ValueError("Se requieren al menos 4 puntos correspondientes en ambas cámaras para calcular la homografía.")

    pts_src = np.array(puntos_src[:4], dtype=np.float32)
    pts_dst = np.array(puntos_dst[:4], dtype=np.float32)

    H, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("No se pudo resolver la homografía. Verifique que los 4 puntos no estén colineales.")

    pts_src_re = pts_src.reshape(-1, 1, 2)
    pts_proj = cv2.perspectiveTransform(pts_src_re, H).reshape(-1, 2)
    errores = np.linalg.norm(pts_proj - pts_dst, axis=1)
    error_medio = float(np.mean(errores))

    return H, error_medio


def calcular_homografia_por_ids_zonas(zonas_cam_sec, zonas_cam_ref):
    """
    Empareja polígonos de zonas con el mismo ID (ej. gondola_1 <-> gondola_1, flujo_1 <-> flujo_1)
    entre la cámara secundaria y la cámara de referencia, extrayendo sus vértices para resolver la homografía.
    """
    dict_ref = {z["id"]: z["polygon"] for z in zonas_cam_ref if "polygon" in z and "id" in z}
    dict_sec = {z["id"]: z["polygon"] for z in zonas_cam_sec if "polygon" in z and "id" in z}

    pts_src = []
    pts_dst = []
    zonas_coincidentes = []

    for zid, poly_sec in dict_sec.items():
        if zid in dict_ref:
            poly_ref = dict_ref[zid]
            min_pts = min(len(poly_sec), len(poly_ref))
            for i in range(min_pts):
                pts_src.append(poly_sec[i])
                pts_dst.append(poly_ref[i])
            zonas_coincidentes.append(zid)

    if len(pts_src) < 4:
        raise ValueError(
            f"Vértices de zonas coincidentes insuficientes ({len(pts_src)} puntos).\n"
            f"Se requieren al menos 4 vértices en zonas con IDs idénticos (ej. 'gondola_1', 'gondola_2').\n"
            f"Zonas encontradas con el mismo ID: {zonas_coincidentes}"
        )

    H, error_medio = calcular_homografia(pts_src, pts_dst)
    return H, error_medio, zonas_coincidentes


def calcular_homografia_por_emparejamiento_zonas(zonas_cam_sec, zonas_cam_ref, mapeo_pares):
    """
    Empareja polígonos de zonas según una lista de pares explícitos seleccionados por el usuario:
    mapeo_pares = [ (id_sec_1, id_ref_1), (id_sec_2, id_ref_2), ... ]
    Extrae los vértices correspondientes de cada par para calcular la homografía.
    """
    dict_ref = {z["id"]: z["polygon"] for z in zonas_cam_ref if "polygon" in z and "id" in z}
    dict_sec = {z["id"]: z["polygon"] for z in zonas_cam_sec if "polygon" in z and "id" in z}

    pts_src = []
    pts_dst = []
    pares_procesados = []

    for id_sec, id_ref in mapeo_pares:
        if id_sec in dict_sec and id_ref in dict_ref:
            poly_sec = dict_sec[id_sec]
            poly_ref = dict_ref[id_ref]
            min_pts = min(len(poly_sec), len(poly_ref))
            for i in range(min_pts):
                pts_src.append(poly_sec[i])
                pts_dst.append(poly_ref[i])
            pares_procesados.append((id_sec, id_ref))

    if len(pts_src) < 4:
        raise ValueError(
            f"Vértices de góndolas emparejadas insuficientes ({len(pts_src)} puntos).\n"
            f"Se requieren al menos 4 vértices en las góndolas/zonas emparejadas.\n"
            f"Pares emparejados válidos: {pares_procesados}"
        )

    pts_src_np = np.array(pts_src, dtype=np.float32)
    pts_dst_np = np.array(pts_dst, dtype=np.float32)

    H, status = cv2.findHomography(pts_src_np, pts_dst_np, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("No se pudo resolver la homografía con los vértices de las góndolas emparejadas.")

    pts_src_re = pts_src_np.reshape(-1, 1, 2)
    pts_proj = cv2.perspectiveTransform(pts_src_re, H).reshape(-1, 2)
    errores = np.linalg.norm(pts_proj - pts_dst_np, axis=1)
    error_medio = float(np.mean(errores))

    return H, error_medio, pares_procesados



def transformar_puntos_parquet(df, H):
    """
    Aplica la matriz de Homografía H a las coordenadas (x_foot, y_foot) de un DataFrame de tracking.
    """
    if len(df) == 0 or H is None:
        return df

    df_res = df.copy()
    pts = df_res[["x_foot", "y_foot"]].values.astype(np.float32).reshape(-1, 1, 2)
    pts_trans = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    df_res["x_foot_ref"] = pts_trans[:, 0]
    df_res["y_foot_ref"] = pts_trans[:, 1]
    return df_res


def crear_fondo_plano_tienda_2d(w=1200, h=800):
    """
    Crea un fondo esquemático del plano 2D del piso del local comercial (Vista Cenital / Top-Down).
    """
    canvas = np.full((h, w, 3), (24, 25, 38), dtype=np.uint8)

    # Cuadrícula suave de piso (Grid de 50px)
    for x in range(0, w, 50):
        cv2.line(canvas, (x, 0), (x, h), (38, 40, 58), 1)
    for y in range(0, h, 50):
        cv2.line(canvas, (0, y), (w, y), (38, 40, 58), 1)

    # Borde exterior del local
    cv2.rectangle(canvas, (20, 20), (w - 20, h - 20), (137, 180, 250), 2)
    cv2.putText(canvas, "PLANO 2D CENITAL DE LA TIENDA (TOP-DOWN BIRDS-EYE VIEW)", (35, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (137, 180, 250), 2)

    return canvas


def generar_mapa_calor_planta_2d_topdown(dfs_dict, homografias_plano_dict, w_plano=1200, h_plano=800, carpeta_salida="homografia"):
    """
    Proyecta el seguimiento de todas las cámaras sobre un Plano 2D Cenital (Top-Down) libre de deformaciones.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    fondo_plano = crear_fondo_plano_tienda_2d(w=w_plano, h=h_plano)

    dfs_transformados = {}

    for cam_id, df_cam in dfs_dict.items():
        if cam_id in homografias_plano_dict:
            H = np.array(homografias_plano_dict[cam_id])
            df_trans = transformar_puntos_parquet(df_cam, H)
            dfs_transformados[cam_id] = df_trans
        else:
            df_c = df_cam.copy()
            df_c["x_foot_ref"] = df_c["x_foot"]
            df_c["y_foot_ref"] = df_c["y_foot"]
            dfs_transformados[cam_id] = df_c

    if not dfs_transformados:
        return None

    list_dfs = list(dfs_transformados.values())
    df_total = pd.concat(list_dfs, ignore_index=True)
    df_validos = df_total.dropna(subset=["x_foot_ref", "y_foot_ref"]).copy()

    df_validos = df_validos[
        (df_validos["x_foot_ref"] >= 0) & (df_validos["x_foot_ref"] < w_plano) &
        (df_validos["y_foot_ref"] >= 0) & (df_validos["y_foot_ref"] < h_plano)
    ]

    heatmap_acc = np.zeros((h_plano, w_plano), dtype=np.float32)
    for _, row in df_validos.iterrows():
        xi = int(row["x_foot_ref"])
        yi = int(row["y_foot_ref"])
        if 0 <= xi < w_plano and 0 <= yi < h_plano:
            heatmap_acc[yi, xi] += 1.0

    sigma = 35
    suavizado = cv2.GaussianBlur(heatmap_acc, (0, 0), sigmaX=sigma, sigmaY=sigma)
    resultado = fondo_plano.copy()

    if suavizado.max() > 0:
        pct_high = np.percentile(suavizado[suavizado > 0], 97) if np.any(suavizado > 0) else 1.0
        norm_pow = np.power(suavizado / max(pct_high, 0.001), 0.45)
        normalizado = np.clip(norm_pow * 255.0, 0, 255).astype(np.uint8)

        color = cv2.applyColorMap(normalizado, cv2.COLORMAP_JET)
        mascara = normalizado > 15
        mezcla = cv2.addWeighted(fondo_plano, 0.35, color, 0.65, 0)
        resultado[mascara] = mezcla[mascara]

    salida_png = os.path.join(carpeta_salida, "mapa_calor_planta_2d_topdown.png")
    cv2.imwrite(salida_png, resultado)
    print(f"✅ Mapa de calor Top-Down 2D guardado en: {salida_png}")

    # Trayectorias Top-Down 2D por cámara
    vis_tray = fondo_plano.copy()
    legend_x = 40
    for idx_c, (cid, df_c) in enumerate(dfs_transformados.items()):
        color_cam = COLORES_CAMARAS[idx_c % len(COLORES_CAMARAS)]
        df_c_val = df_c.dropna(subset=["x_foot_ref", "y_foot_ref"])

        for _, row in df_c_val.iloc[::2].iterrows():
            xf = int(row["x_foot_ref"])
            yf = int(row["y_foot_ref"])
            if 0 <= xf < w_plano and 0 <= yf < h_plano:
                cv2.circle(vis_tray, (xf, yf), 4, color_cam, -1)

        cv2.circle(vis_tray, (legend_x, h_plano - 35), 8, color_cam, -1)
        cv2.putText(vis_tray, f"Cámara: {cid}", (legend_x + 15, h_plano - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        legend_x += 280

    salida_tray = os.path.join(carpeta_salida, "verificacion_trayectorias_planta_2d.png")
    cv2.imwrite(salida_tray, vis_tray)

    salida_parquet = os.path.join(carpeta_salida, "tracking_planta_2d_unificado.parquet")
    df_validos.to_parquet(salida_parquet, index=False)

    return salida_png


def generar_mapa_calor_multicamara_fusionado(dfs_dict, homografias_dict, cam_ref_id, fondo_ref, carpeta_salida="homografia"):
    """
    Genera la suite completa de entregables y comprobaciones visuales de Homografía.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    h_ref, w_ref = fondo_ref.shape[:2]

    dfs_transformados = {}

    for cam_id, df_cam in dfs_dict.items():
        if cam_id == cam_ref_id:
            df_c = df_cam.copy()
            df_c["x_foot_ref"] = df_c["x_foot"]
            df_c["y_foot_ref"] = df_c["y_foot"]
            dfs_transformados[cam_id] = df_c
        elif cam_id in homografias_dict:
            H = np.array(homografias_dict[cam_id])
            df_trans = transformar_puntos_parquet(df_cam, H)
            dfs_transformados[cam_id] = df_trans

    if not dfs_transformados:
        return None

    list_dfs = list(dfs_transformados.values())
    df_total = pd.concat(list_dfs, ignore_index=True)
    df_validos = df_total.dropna(subset=["x_foot_ref", "y_foot_ref"]).copy()

    df_validos = df_validos[
        (df_validos["x_foot_ref"] >= 0) & (df_validos["x_foot_ref"] < w_ref) &
        (df_validos["y_foot_ref"] >= 0) & (df_validos["y_foot_ref"] < h_ref)
    ]

    heatmap_acc = np.zeros((h_ref, w_ref), dtype=np.float32)
    for _, row in df_validos.iterrows():
        xi = int(row["x_foot_ref"])
        yi = int(row["y_foot_ref"])
        if 0 <= xi < w_ref and 0 <= yi < h_ref:
            heatmap_acc[yi, xi] += 1.0

    sigma = 30
    suavizado = cv2.GaussianBlur(heatmap_acc, (0, 0), sigmaX=sigma, sigmaY=sigma)
    resultado = fondo_ref.copy()

    if suavizado.max() > 0:
        pct_high = np.percentile(suavizado[suavizado > 0], 97) if np.any(suavizado > 0) else 1.0
        norm_pow = np.power(suavizado / max(pct_high, 0.001), 0.45)
        normalizado = np.clip(norm_pow * 255.0, 0, 255).astype(np.uint8)

        color = cv2.applyColorMap(normalizado, cv2.COLORMAP_JET)
        mascara = normalizado > 15
        mezcla = cv2.addWeighted(fondo_ref, 0.30, color, 0.70, 0)
        resultado[mascara] = mezcla[mascara]

    n_cams = len(dfs_transformados)
    n_det = len(df_validos)

    cv2.rectangle(resultado, (0, 0), (w_ref, 42), (15, 15, 15), -1)
    cv2.putText(resultado, f"MAPA DE CALOR MULTICÁMARA FUSIONADO (NORMALIZADO) | {n_cams} Cámaras | {n_det} Puntos",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)

    salida_png = os.path.join(carpeta_salida, "mapa_calor_multicamara_fusionado.png")
    cv2.imwrite(salida_png, resultado)

    vis_trayectorias = fondo_ref.copy()
    cv2.rectangle(vis_trayectorias, (0, 0), (w_ref, 42), (15, 15, 15), -1)
    cv2.putText(vis_trayectorias, "VERIFICACIÓN VISUAL DE TRAYECTORIAS Y POSICIONES PROYECTADAS (2D)",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    legend_x = 20
    for idx_c, (cid, df_c) in enumerate(dfs_transformados.items()):
        color_cam = COLORES_CAMARAS[idx_c % len(COLORES_CAMARAS)]
        df_c_val = df_c.dropna(subset=["x_foot_ref", "y_foot_ref"])

        for _, row in df_c_val.iloc[::3].iterrows():
            xf = int(row["x_foot_ref"])
            yf = int(row["y_foot_ref"])
            if 0 <= xf < w_ref and 0 <= yf < h_ref:
                cv2.circle(vis_trayectorias, (xf, yf), 3, color_cam, -1)

        etiqueta = f"🔵 Ref: {cid}" if cid == cam_ref_id else f"🟢 Transformed: {cid}"
        cv2.circle(vis_trayectorias, (legend_x, h_ref - 25), 8, color_cam, -1)
        cv2.putText(vis_trayectorias, etiqueta, (legend_x + 15, h_ref - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        legend_x += 260

    salida_trayectorias = os.path.join(carpeta_salida, "verificacion_trayectorias_multicamara.png")
    cv2.imwrite(salida_trayectorias, vis_trayectorias)

    salida_parquet = os.path.join(carpeta_salida, "tracking_multicamara_unificado.parquet")
    df_validos.to_parquet(salida_parquet, index=False)

    generar_mapa_calor_planta_2d_topdown(dfs_dict, homografias_dict, carpeta_salida=carpeta_salida)

    return salida_png


def generar_vista_2d_superpuesta_warped(frame_ref, frame_sec, H, carpeta_salida="homografia"):
    """
    Genera la imagen de comprobación óptica 2D: aplica cv2.warpPerspective al frame de la cámara
    secundaria usando la matriz H y lo mezcla en un 50% con el frame de la cámara de referencia.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    h_ref, w_ref = frame_ref.shape[:2]

    frame_sec_warped = cv2.warpPerspective(frame_sec, H, (w_ref, h_ref))
    superpuesta = cv2.addWeighted(frame_ref, 0.50, frame_sec_warped, 0.50, 0)

    cv2.rectangle(superpuesta, (0, 0), (w_ref, 45), (15, 15, 15), -1)
    cv2.putText(superpuesta, "COMPROBACIÓN ÓPTICA 2D: SUPERPOSICIÓN DE PERSPECTIVAS (WARPED HOMOGRAPHY)",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 200), 2)

    salida_warped = os.path.join(carpeta_salida, "verificacion_superposicion_optica_2d.png")
    cv2.imwrite(salida_warped, superpuesta)
    return salida_warped
