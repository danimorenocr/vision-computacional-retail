# Crear entorno virtual
python -m venv .venv

# Activar el entorno virtual
.venv\Scripts\Activate.ps1  

# Dependencias
pip install -r .\requirements.txt  

# Correr app local
python app_retail_tracker.py

# Cortar Clips
python cortar_clip.py --video grabacion3.mp4 --inicio 2:20 --fin 2:45 --target_fps 20 --salida clip_test.mp4

⚙️ Parámetros de cortar_clip.py:
--video: Video de origen.
--inicio: Tiempo de inicio (ej: 0, 45, 1:30, 02:15).
--fin: Tiempo de fin (ej: 90, 3:00, 05:00).
--duracion: Duración en segundos a partir del inicio.
--target_fps: FPS deseados de salida (ej: 20 o 25 para reducir los 60 FPS y acelerar YOLO 3x).
--salida: Nombre del archivo de salida (por defecto clip_test.mp4).

# Extraer datos de personas en el video
python 1_extraer_datos.py --modo live --video clip_test.mp4 --camera_id cam_01 --frame_skip 2
python 1_extraer_datos.py --modo guardar --video clip_test.mp4 --camera_id cam_01 --frame_skip 2

python 2_definir_zonas.py --video clip_test.mp4 --camera_id cam_01
  Controles de Teclado en 2_definir_zonas.py:
  - 'f' : Zona de Pasillo / Flujo (Verde)
  - 'i' : Zona de Góndola / Interacción (Naranja)
  - 'e' : Zona de EXCLUSIÓN / IGNORAR (Rojo - Elimina 100% las detecciones de esa área)
  - 'n' : Cancelar polígono en curso
  - 's' : Guardar y salir

python 3_procesar_zonas.py --parquet datos_parquet/tracking_cam_01.parquet --zonas zonas_cam_01.json --video clip_test.mp4

python generar_mapa_calor.py --parquet datos_parquet/tracking_cam_01.parquet --video clip_test.mp4 --zonas zonas_cam_01.json


python anotar_video.py --video clip_test.mp4 --parquet datos_parquet/tracking_cam_01.parquet --zonas zonas_cam_01.json
