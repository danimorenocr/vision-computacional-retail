"""
===============================================================================
🛒 RETAIL TRACKER GUI - APLICACIÓN UNIFICADA DE EXTRACCIÓN Y ANALÍTICA RETAIL
===============================================================================
Interfaz local moderna (Tkinter + OpenCV + Pillow) que unifica:
1. Pestaña 1: Selección de video, extracción YOLO Pose y Editor de Zonas.
2. Pestaña 2: Calibración Guiada de Homografía Lado a Lado con Carga Directa
   de JSONs de Zonas (Góndolas y Pasillos) y representación en Canvases.
3. Checkbox opcional para activar/desactivar el renderizado del video anotado.
===============================================================================
"""

import os
import sys
import cv2
import json
import glob
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
from PIL import Image, ImageTk

# Importar funciones de la pipeline local
import importlib.util

def cargar_modulo(nombre_modulo, ruta_archivo):
    spec = importlib.util.spec_from_file_location(nombre_modulo, ruta_archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

dir_actual = os.path.dirname(os.path.abspath(__file__))
extraer_mod = cargar_modulo("extraer_mod", os.path.join(dir_actual, "1_extraer_datos.py"))
procesar_mod = cargar_modulo("procesar_mod", os.path.join(dir_actual, "3_procesar_zonas.py"))
mapas_mod = cargar_modulo("mapas_mod", os.path.join(dir_actual, "generar_mapa_calor.py"))
anotar_mod = cargar_modulo("anotar_mod", os.path.join(dir_actual, "anotar_video.py"))
fusion_mod = cargar_modulo("fusion_mod", os.path.join(dir_actual, "fusion_multicamara.py"))


class RetailTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🛒 RetailTracker Studio - Calibración Guiada de Homografía Lado a Lado por Zonas JSON")
        self.root.geometry("1580x940")
        self.root.minsize(1280, 720)

        # Tema Oscuro Premium
        self.colors = {
            "bg": "#181825",
            "panel": "#1e1e2e",
            "card": "#2b2b3b",
            "fg": "#cdd6f4",
            "accent": "#89b4fa",
            "success": "#a6e3a1",
            "warning": "#fab387",
            "danger": "#f38ba8",
            "flujo": "#00FF00",
            "interaccion": "#FF9900",
            "exclusion": "#FF0000"
        }

        self.root.configure(bg=self.colors["bg"])
        self.setup_styles()

        # Variables Pestaña 1
        self.video_path = tk.StringVar()
        self.camera_id = tk.StringVar(value="cam_04")
        self.carpeta_salida = tk.StringVar()
        self.modelo_path = tk.StringVar(value="yolo26n-pose_640_openvino_model")
        self.imgsz = tk.IntVar(value=640)
        self.frame_skip = tk.IntVar(value=1)
        self.conf_thresh = tk.DoubleVar(value=0.25)
        self.dispositivo_opt = tk.StringVar(value="⚡ GPU (CUDA)" if torch.cuda.is_available() else "💻 Solo CPU")
        self.generar_video_anotado = tk.BooleanVar(value=False)

        self.frame_original = None
        self.h_orig = 0
        self.w_orig = 0
        self.scale_x = 1.0
        self.scale_y = 1.0

        self.zonas = []
        self.poligono_actual = []
        self.contador_zonas = {"flujo": 0, "interaccion": 0, "exclusion": 0}

        # Variables Pestaña 2: Calibración Guiada Lado a Lado
        self.cam_ref_id = tk.StringVar(value="cam_04")
        self.cam_ref_video = tk.StringVar()
        self.parquet_ref_path = tk.StringVar()
        self.json_ref_path = tk.StringVar()
        self.zonas_hom_ref = []

        self.cam_sec_id = tk.StringVar(value="cam_05")
        self.cam_sec_video = tk.StringVar()
        self.parquet_sec_path = tk.StringVar()
        self.json_sec_path = tk.StringVar()
        self.zonas_hom_sec = []

        self.frame_ref = None
        self.w_ref, self.h_ref = 0, 0
        self.scale_ref_x, self.scale_ref_y = 1.0, 1.0

        self.frame_sec = None
        self.w_sec, self.h_sec = 0, 0
        self.scale_sec_x, self.scale_sec_y = 1.0, 1.0

        self.puntos_hom_ref = []
        self.puntos_hom_sec = []
        self.turno_homografía = "ref"
        self.homografias = {}
        self.emparejamiento_zonas = []  # Lista de tuplas: [(id_sec, id_ref), ...]

        self.tracking_running = False
        self.tracking_complete = False
        self.log_queue = queue.Queue()

        self.build_gui()
        self.escaneo_inicial_videos()
        self.root.after(100, self.procesar_mensajes_queue)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.colors["panel"])
        style.configure("Card.TFrame", background=self.colors["card"], relief="flat")
        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=self.colors["panel"], foreground=self.colors["fg"], padding=[15, 8], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", self.colors["accent"])], foreground=[("selected", "#11111b")])

        style.configure("TLabel", background=self.colors["panel"], foreground=self.colors["fg"], font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=self.colors["panel"], foreground=self.colors["accent"], font=("Segoe UI", 12, "bold"))
        style.configure("Title.TLabel", background=self.colors["panel"], foreground=self.colors["fg"], font=("Segoe UI", 14, "bold"))

        style.configure("TButton", background=self.colors["card"], foreground=self.colors["fg"], borderwidth=0, font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", self.colors["accent"])], foreground=[("active", "#11111b")])

        style.configure("Accent.TButton", background=self.colors["accent"], foreground="#11111b", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#b4befe")])

        style.configure("Flujo.TButton", background="#059669", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.map("Flujo.TButton", background=[("active", "#10b981")])

        style.configure("Gondola.TButton", background="#d97706", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.map("Gondola.TButton", background=[("active", "#f59e0b")])

        style.configure("Exclusion.TButton", background="#dc2626", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.map("Exclusion.TButton", background=[("active", "#ef4444")])

        style.configure("TProgressbar", thickness=14, troughcolor=self.colors["card"], background=self.colors["accent"])

    def build_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---------------------------------------------------------------------
        # PESTAÑA 1: INFERENCIA Y EDITOR DE ZONAS
        # ---------------------------------------------------------------------
        self.tab_editor = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_editor, text="🛒 1. Inferencia y Zonas")

        main_container = ttk.Frame(self.tab_editor)
        main_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_panel = ttk.Frame(main_container, width=460)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        lbl_title = ttk.Label(left_panel, text="🛒 RetailTracker Studio", style="Title.TLabel")
        lbl_title.pack(anchor="w", pady=(0, 2))

        lbl_sub = ttk.Label(left_panel, text="Inferencia + Editor de Zonas + Analítica", font=("Segoe UI", 9))
        lbl_sub.pack(anchor="w", pady=(0, 10))

        sec1 = ttk.Frame(left_panel, style="Card.TFrame", padding=10)
        sec1.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(sec1, text="1. Configuración de Entrada y Salida", style="Header.TLabel", background=self.colors["card"]).pack(anchor="w", pady=(0, 6))

        ttk.Label(sec1, text="Seleccionar Video:", background=self.colors["card"]).pack(anchor="w")
        frame_combo = ttk.Frame(sec1, style="Card.TFrame")
        frame_combo.pack(fill=tk.X, pady=(2, 6))

        self.cb_videos = ttk.Combobox(frame_combo, textvariable=self.video_path, font=("Segoe UI", 9))
        self.cb_videos.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.cb_videos.bind("<<ComboboxSelected>>", self.on_video_selected)

        btn_browse = ttk.Button(frame_combo, text="📁 Buscar", width=8, command=self.buscar_video)
        btn_browse.pack(side=tk.RIGHT)

        ttk.Label(sec1, text="ID de Cámara Actual:", background=self.colors["card"]).pack(anchor="w")
        ent_cam = ttk.Entry(sec1, textvariable=self.camera_id, font=("Segoe UI", 9))
        ent_cam.pack(fill=tk.X, pady=(2, 6))
        self.camera_id.trace_add("write", lambda *args: self.actualizar_carpeta_salida())

        ttk.Label(sec1, text="Modelo Pose (PyTorch / OpenVINO):", background=self.colors["card"]).pack(anchor="w")
        frame_modelo = ttk.Frame(sec1, style="Card.TFrame")
        frame_modelo.pack(fill=tk.X, pady=(2, 6))

        self.modelos_disponibles = [
            "yolo26n-pose_640_openvino_model",
            "yolo26n-pose_480_openvino_model",
            "yolo26n-pose.pt",
            "yolo26s-pose.pt"
        ]

        self.cb_modelo = ttk.Combobox(frame_modelo, textvariable=self.modelo_path, values=self.modelos_disponibles, font=("Segoe UI", 9))
        self.cb_modelo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.cb_modelo.bind("<<ComboboxSelected>>", self.on_modelo_selected)

        btn_browse_mod = ttk.Button(frame_modelo, text="📁 Buscar", width=8, command=self.buscar_modelo)
        btn_browse_mod.pack(side=tk.RIGHT)

        ttk.Label(sec1, text="Carpeta de Salida Unificada:", background=self.colors["card"]).pack(anchor="w")
        ent_out = ttk.Entry(sec1, textvariable=self.carpeta_salida, font=("Segoe UI", 9))
        ent_out.pack(fill=tk.X, pady=(2, 6))

        frame_params = ttk.Frame(sec1, style="Card.TFrame")
        frame_params.pack(fill=tk.X, pady=(4, 0))

        lbl_fskip = ttk.Label(frame_params, text="Frame Skip:", background=self.colors["card"])
        lbl_fskip.grid(row=0, column=0, sticky="w", padx=(0, 3))
        spn_fskip = ttk.Spinbox(frame_params, from_=1, to=10, textvariable=self.frame_skip, width=4)
        spn_fskip.grid(row=0, column=1, padx=(0, 8))

        lbl_conf = ttk.Label(frame_params, text="Confianza:", background=self.colors["card"])
        lbl_conf.grid(row=0, column=2, sticky="w", padx=(0, 3))
        spn_conf = ttk.Spinbox(frame_params, from_=0.1, to=0.9, increment=0.05, textvariable=self.conf_thresh, width=4)
        spn_conf.grid(row=0, column=3, padx=(0, 8))

        lbl_imgsz = ttk.Label(frame_params, text="ImgSize:", background=self.colors["card"])
        lbl_imgsz.grid(row=0, column=4, sticky="w", padx=(0, 3))
        spn_imgsz = ttk.Spinbox(frame_params, values=(480, 640, 320, 1280), textvariable=self.imgsz, width=5)
        spn_imgsz.grid(row=0, column=5)

        lbl_disp = ttk.Label(frame_params, text="⚡ Motor:", background=self.colors["card"])
        lbl_disp.grid(row=1, column=0, columnspan=2, sticky="w", padx=(0, 3), pady=(5, 0))
        cb_disp = ttk.Combobox(frame_params, textvariable=self.dispositivo_opt, values=["⚡ GPU (CUDA)", "💻 Solo CPU"], state="readonly", width=14, font=("Segoe UI", 9))
        cb_disp.grid(row=1, column=2, columnspan=4, sticky="w", pady=(5, 0))

        sec2 = ttk.Frame(left_panel, style="Card.TFrame", padding=10)
        sec2.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(sec2, text="2. Iniciar Inferencia (Paso 1)", style="Header.TLabel", background=self.colors["card"]).pack(anchor="w", pady=(0, 6))

        self.btn_run_tracking = ttk.Button(sec2, text="🚀 INICIAR EXTRACCIÓN Y DIBUJAR ZONAS", style="Accent.TButton", command=self.iniciar_pipeline)
        self.btn_run_tracking.pack(fill=tk.X, ipady=5, pady=(0, 6))

        self.progress_bar = ttk.Progressbar(sec2, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(0, 4))

        self.lbl_status = ttk.Label(sec2, text="Estado: En espera de video...", background=self.colors["card"], font=("Segoe UI", 9, "italic"))
        self.lbl_status.pack(anchor="w")

        sec3 = ttk.Frame(left_panel, style="Card.TFrame", padding=10)
        sec3.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(sec3, text="3. Opciones de Salida y Generación", style="Header.TLabel", background=self.colors["card"]).pack(anchor="w", pady=(0, 6))

        self.chk_anotado = tk.Checkbutton(
            sec3, text="🎥 Generar Video Anotado (MP4)",
            variable=self.generar_video_anotado,
            bg=self.colors["card"], fg=self.colors["fg"],
            selectcolor=self.colors["panel"],
            activebackground=self.colors["card"], activeforeground=self.colors["fg"],
            font=("Segoe UI", 9, "bold")
        )
        self.chk_anotado.pack(anchor="w", pady=(0, 6))

        self.btn_generar_mapas = ttk.Button(sec3, text="📊 GENERAR MAPAS Y MÉTRICAS", command=self.generar_mapas_y_metricas)
        self.btn_generar_mapas.pack(fill=tk.X, ipady=4)

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        canvas_bar = ttk.Frame(right_panel)
        canvas_bar.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(canvas_bar, text="📍 Editor Visual de Zonas:", style="Header.TLabel").pack(side=tk.LEFT, padx=(0, 15))

        btn_flujo = ttk.Button(canvas_bar, text="🟢 PASILLO ('f')", style="Flujo.TButton", command=lambda: self.guardar_poligono_actual("flujo"))
        btn_flujo.pack(side=tk.LEFT, padx=3)

        btn_gondola = ttk.Button(canvas_bar, text="🟠 GÓNDOLA ('i')", style="Gondola.TButton", command=lambda: self.guardar_poligono_actual("interaccion"))
        btn_gondola.pack(side=tk.LEFT, padx=3)

        btn_excl = ttk.Button(canvas_bar, text="🔴 EXCLUSIÓN ('e')", style="Exclusion.TButton", command=lambda: self.guardar_poligono_actual("exclusion"))
        btn_excl.pack(side=tk.LEFT, padx=3)

        btn_cancel = ttk.Button(canvas_bar, text="❌ Cancelar ('n')", command=self.cancelar_poligono)
        btn_cancel.pack(side=tk.LEFT, padx=8)

        btn_load_json = ttk.Button(canvas_bar, text="📂 Cargar JSON Zonas", command=self.buscar_y_cargar_json_zonas)
        btn_load_json.pack(side=tk.RIGHT, padx=(0, 5))

        btn_save_json = ttk.Button(canvas_bar, text="💾 Guardar JSON Zonas", style="Accent.TButton", command=self.guardar_json_zonas)
        btn_save_json.pack(side=tk.RIGHT)

        work_area = ttk.Frame(right_panel)
        work_area.pack(fill=tk.BOTH, expand=True)

        self.canvas_frame = ttk.Frame(work_area, style="Card.TFrame")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.canvas = tk.Canvas(self.canvas_frame, bg="#0d0e15", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.root.bind("<Key>", self.on_key_press)

        zone_list_frame = ttk.Frame(work_area, width=320, style="Card.TFrame", padding=10)
        zone_list_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        zone_list_frame.pack_propagate(False)

        ttk.Label(zone_list_frame, text="Zonas Creadas:", background=self.colors["card"], font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.lst_zonas = tk.Listbox(zone_list_frame, bg="#181825", fg=self.colors["fg"], selectbackground=self.colors["accent"], selectforeground="#11111b", borderwidth=0, font=("Segoe UI", 9), height=5)
        self.lst_zonas.pack(fill=tk.X, pady=(0, 6))

        frame_zone_btns = ttk.Frame(zone_list_frame, style="Card.TFrame")
        frame_zone_btns.pack(fill=tk.X, pady=(0, 8))

        btn_del_zona = ttk.Button(frame_zone_btns, text="🗑️ Eliminar Zona", command=self.eliminar_zona_seleccionada)
        btn_del_zona.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        btn_clear_zonas = ttk.Button(frame_zone_btns, text="⚠️ Borrar Todas", command=self.limpiar_todas_las_zonas)
        btn_clear_zonas.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))

        ttk.Label(zone_list_frame, text="📋 Registro de Ejecución (Logs):", background=self.colors["card"], font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 4))
        self.txt_log = scrolledtext.ScrolledText(zone_list_frame, bg="#11111b", fg="#a6adc8", font=("Consolas", 8), relief="flat")
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # ---------------------------------------------------------------------
        # PESTAÑA 2: CALIBRACIÓN DE HOMOGRAFÍA LADO A LADO POR ZONAS JSON
        # ---------------------------------------------------------------------
        self.tab_homografía = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_homografía, text="🌐 2. Calibración de Homografía (Lado a Lado)")

        hom_container = ttk.Frame(self.tab_homografía, padding=8)
        hom_container.pack(fill=tk.BOTH, expand=True)

        bar_hom = ttk.Frame(hom_container, style="Card.TFrame", padding=10)
        bar_hom.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(bar_hom, text="🌐 Calibración Multicámara por Puntos P1..P4 (Lado a Lado)", style="Header.TLabel", background=self.colors["card"]).pack(anchor="w", pady=(0, 2))

        self.lbl_instrucciones_hom = ttk.Label(
            bar_hom,
            text="Marca los puntos correspondientes P1..P4 en ambas cámaras para calibrar.",
            background=self.colors["card"], font=("Segoe UI", 10, "bold"), foreground=self.colors["warning"]
        )
        self.lbl_instrucciones_hom.pack(anchor="w", pady=(0, 6))

        frame_cams_select = ttk.Frame(bar_hom, style="Card.TFrame")
        frame_cams_select.pack(fill=tk.X)

        # Fila 1: Entradas para Cámara Referencia (Izquierda)
        ttk.Label(frame_cams_select, text="Cam Ref:", background=self.colors["card"], font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 2))
        ent_ref_id = ttk.Entry(frame_cams_select, textvariable=self.cam_ref_id, width=8)
        ent_ref_id.grid(row=0, column=1, padx=(0, 4))

        btn_browse_ref = ttk.Button(frame_cams_select, text="📁 Video Ref", command=self.cargar_video_hom_ref)
        btn_browse_ref.grid(row=0, column=2, padx=(0, 4))

        btn_parquet_ref = ttk.Button(frame_cams_select, text="📊 Parquet Ref", command=self.buscar_parquet_ref)
        btn_parquet_ref.grid(row=0, column=3, padx=(0, 15))

        # Fila 1: Entradas para Cámara Secundaria (Derecha)
        ttk.Label(frame_cams_select, text="Cam Sec:", background=self.colors["card"], font=("Segoe UI", 9, "bold")).grid(row=0, column=4, sticky="w", padx=(0, 2))
        ent_sec_id = ttk.Entry(frame_cams_select, textvariable=self.cam_sec_id, width=8)
        ent_sec_id.grid(row=0, column=5, padx=(0, 4))

        btn_browse_sec = ttk.Button(frame_cams_select, text="📁 Video Sec", command=self.cargar_video_hom_sec)
        btn_browse_sec.grid(row=0, column=6, padx=(0, 4))

        btn_parquet_sec = ttk.Button(frame_cams_select, text="📊 Parquet Sec", command=self.buscar_parquet_sec)
        btn_parquet_sec.grid(row=0, column=7, padx=(0, 15))

        # Acciones de Homografía
        btn_calc_hom = ttk.Button(frame_cams_select, text="📐 GENERAR MAPAS EN 'homografia/'", style="Accent.TButton", command=self.resolver_homografia_y_guardar_carpeta_homografia)
        btn_calc_hom.grid(row=0, column=8, padx=(0, 0))

        # Canvases Lado a Lado
        canvases_hom_frame = ttk.Frame(hom_container)
        canvases_hom_frame.pack(fill=tk.BOTH, expand=True)

        box_left = ttk.Frame(canvases_hom_frame, style="Card.TFrame", padding=6)
        box_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.lbl_ref_title = ttk.Label(box_left, text="Cámara Referencia (Izquierda - Zonas & Puntos)", style="Header.TLabel", background=self.colors["card"])
        self.lbl_ref_title.pack(anchor="w", pady=(0, 4))

        self.canvas_hom_ref = tk.Canvas(box_left, bg="#0d0e15", highlightthickness=0)
        self.canvas_hom_ref.pack(fill=tk.BOTH, expand=True)
        self.canvas_hom_ref.bind("<Button-1>", self.on_canvas_ref_click)

        box_right = ttk.Frame(canvases_hom_frame, style="Card.TFrame", padding=6)
        box_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.lbl_sec_title = ttk.Label(box_right, text="Cámara Secundaria (Derecha - Zonas & Puntos)", style="Header.TLabel", background=self.colors["card"])
        self.lbl_sec_title.pack(anchor="w", pady=(0, 4))

        self.canvas_hom_sec = tk.Canvas(box_right, bg="#0d0e15", highlightthickness=0)
        self.canvas_hom_sec.pack(fill=tk.BOTH, expand=True)
        self.canvas_hom_sec.bind("<Button-1>", self.on_canvas_sec_click)

        # Leyenda de Colores
        bar_legend = ttk.Frame(hom_container, style="Card.TFrame", padding=8)
        bar_legend.pack(fill=tk.X, pady=(6, 0))

        lbl_leyenda = ttk.Label(
            bar_legend,
            text="Calibración: 🏷️ Carga JSONs de zonas con IDs idénticos (ej. 'gondola_1', 'gondola_2') o marca 🟣 P1  🟢 P2  🟠 P3  🟡 P4",
            font=("Segoe UI", 10, "bold"), foreground=self.colors["accent"], background=self.colors["card"]
        )
        lbl_leyenda.pack(side=tk.LEFT)

        btn_undo_pt = ttk.Button(bar_legend, text="↩️ Deshacer Último Punto", command=self.deshacer_ultimo_punto_homografia)
        btn_undo_pt.pack(side=tk.RIGHT, padx=6)

        btn_clear_pts = ttk.Button(bar_legend, text="⚠️ Reiniciar Puntos", command=self.limpiar_puntos_homografia_lado_a_lado)
        btn_clear_pts.pack(side=tk.RIGHT)

    # -------------------------------------------------------------------------
    # FUNCIONES PESTAÑA 1 (INFERENCIA Y ZONAS)
    # -------------------------------------------------------------------------
    def on_modelo_selected(self, event=None):
        mod = self.modelo_path.get()
        if "480" in mod:
            self.imgsz.set(480)
        elif "640" in mod:
            self.imgsz.set(640)
        self.log(f"🧠 Modelo seleccionado: {mod} | ImgSize: {self.imgsz.get()}")

    def buscar_modelo(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar Modelo (.pt o carpeta OpenVINO)",
            filetypes=[("Modelos PyTorch", "*.pt"), ("Todos los archivos", "*.*")]
        )
        if ruta:
            self.modelo_path.set(ruta)
            self.on_modelo_selected()

    def obtener_ruta_modelo_absoluta(self):
        mod = self.modelo_path.get().strip()
        if os.path.isabs(mod) and os.path.exists(mod):
            return mod
        candidato = os.path.join(dir_actual, mod)
        if os.path.exists(candidato):
            return candidato
        return mod

    def escaneo_inicial_videos(self):
        archivos = glob.glob("*.mp4") + glob.glob("*.avi") + glob.glob("*.mkv")
        if archivos:
            self.cb_videos["values"] = archivos
            self.cb_videos.current(0)
            self.on_video_selected(None)

    def buscar_video(self):
        ruta = filedialog.askopenfilename(title="Seleccionar Video", filetypes=[("Videos", "*.mp4 *.avi *.mkv *.mov")])
        if ruta:
            self.video_path.set(ruta)
            self.on_video_selected(None)

    def on_video_selected(self, event):
        vpath = self.video_path.get()
        if not vpath or not os.path.exists(vpath):
            return

        self.actualizar_carpeta_salida()
        self.cargar_primer_frame_video(vpath)

        cam_id = self.camera_id.get().strip() or "cam_04"
        
        # Buscar JSON al mismo nivel del video con su mismo nombre primero
        video_dir = os.path.dirname(os.path.abspath(vpath))
        base_name = os.path.splitext(os.path.basename(vpath))[0]

        posibles_jsons = [
            os.path.join(video_dir, f"{base_name}_zonas.json"),
            os.path.join(video_dir, f"{base_name}.json"),
            os.path.join(self.carpeta_salida.get(), f"zonas_{cam_id}.json"),
            f"zonas_{cam_id}.json"
        ]

        for pjson in posibles_jsons:
            if os.path.exists(pjson):
                self.cargar_json_existente(pjson)
                break

    def actualizar_carpeta_salida(self):
        vpath = self.video_path.get()
        if not vpath:
            return
        base_name = os.path.splitext(os.path.basename(vpath))[0]
        self.carpeta_salida.set(f"salida_{base_name}")

    def cargar_primer_frame_video(self, vpath):
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Error de Video", f"No se pudo leer el video: {vpath}")
            return

        nombre_video = os.path.basename(vpath)
        duracion_s = total_frames / fps if fps > 0 else 0.0

        print("\n" + "=" * 60)
        print(f"🎬 VIDEO SELECCIONADO: {nombre_video}")
        print(f"   ► FPS (Cuadros por segundo): {fps:.2f} fps")
        print(f"   ► Total de frames: {total_frames}")
        print(f"   ► Resolución: {frame.shape[1]}x{frame.shape[0]} px")
        if duracion_s > 0:
            m = int(duracion_s // 60)
            s = int(duracion_s % 60)
            print(f"   ► Duración total: {m:02d}:{s:02d} ({duracion_s:.1f}s)")
        print("=" * 60 + "\n")

        self.frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.h_orig, self.w_orig = self.frame_original.shape[:2]
        self.log(f"📹 Video cargado: {nombre_video} | {fps:.2f} FPS | {self.w_orig}x{self.h_orig} px ({total_frames} frames)")
        self.redibujar_canvas()

    def redibujar_canvas(self):
        if self.frame_original is None:
            return

        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        if c_w < 50 or c_h < 50:
            c_w, c_h = 960, 540

        self.scale_x = c_w / self.w_orig
        self.scale_y = c_h / self.h_orig

        img_resized = cv2.resize(self.frame_original, (c_w, c_h))
        self.photo_img = ImageTk.PhotoImage(Image.fromarray(img_resized))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo_img, anchor=tk.NW)

        for z in self.zonas:
            t = z.get("type", "flujo")
            pts = z["polygon"]
            pts_canvas = [(int(px * self.scale_x), int(py * self.scale_y)) for px, py in pts]
            flat_pts = [c for p in pts_canvas for c in p]

            color_line = "#00FF00" if t == "flujo" else ("#FF9900" if t == "interaccion" else "#FF0000")

            if len(pts_canvas) >= 3:
                self.canvas.create_polygon(flat_pts, outline=color_line, fill="", width=2)

            x0, y0 = pts_canvas[0]
            self.canvas.create_rectangle(x0, y0 - 18, x0 + len(z["id"]) * 8 + 10, y0, fill="#11111b", outline=color_line)
            self.canvas.create_text(x0 + 5, y0 - 9, text=z["id"], fill=color_line, font=("Segoe UI", 9, "bold"), anchor="w")

        if self.poligono_actual:
            pts_cur_canvas = [(int(px * self.scale_x), int(py * self.scale_y)) for px, py in self.poligono_actual]
            for i, p in enumerate(pts_cur_canvas):
                self.canvas.create_oval(p[0]-4, p[1]-4, p[0]+4, p[1]+4, fill="#f38ba8", outline="#ffffff")
                if i > 0:
                    prev = pts_cur_canvas[i-1]
                    self.canvas.create_line(prev[0], prev[1], p[0], p[1], fill="#f38ba8", width=2, dash=(4, 2))

    def on_canvas_click(self, event):
        if self.frame_original is None:
            return

        x_orig = int(event.x / self.scale_x)
        y_orig = int(event.y / self.scale_y)
        x_orig = max(0, min(self.w_orig - 1, x_orig))
        y_orig = max(0, min(self.h_orig - 1, y_orig))

        self.poligono_actual.append([x_orig, y_orig])
        self.redibujar_canvas()

    def on_canvas_right_click(self, event):
        if len(self.poligono_actual) >= 3:
            self.guardar_poligono_actual("interaccion")

    def on_key_press(self, event):
        char = event.char.lower()
        if char == 'f':
            self.guardar_poligono_actual("flujo")
        elif char == 'i':
            self.guardar_poligono_actual("interaccion")
        elif char == 'e':
            self.guardar_poligono_actual("exclusion")
        elif char == 'n':
            self.cancelar_poligono()
        elif char == 's':
            self.guardar_json_zonas()

    def guardar_poligono_actual(self, tipo_zona):
        if len(self.poligono_actual) < 3:
            messagebox.showwarning("Atención", "Se requieren al menos 3 puntos para definir una zona.")
            return

        self.contador_zonas[tipo_zona] += 1
        if tipo_zona == "flujo":
            zid = f"flujo_{self.contador_zonas['flujo']}"
        elif tipo_zona == "interaccion":
            zid = f"gondola_{self.contador_zonas['interaccion']}"
        else:
            zid = f"exclusion_{self.contador_zonas['exclusion']}"

        nueva_zona = {"id": zid, "type": tipo_zona, "polygon": self.poligono_actual.copy()}
        self.zonas.append(nueva_zona)
        self.poligono_actual = []

        self.actualizar_lista_box_zonas()
        self.redibujar_canvas()
        self.log(f"✅ Zona agregada [{tipo_zona.upper()}]: {zid}")

    def cancelar_poligono(self):
        self.poligono_actual = []
        self.redibujar_canvas()

    def actualizar_lista_box_zonas(self):
        self.lst_zonas.delete(0, tk.END)
        for z in self.zonas:
            t_icon = "🟢" if z["type"] == "flujo" else ("🟠" if z["type"] == "interaccion" else "🔴")
            self.lst_zonas.insert(tk.END, f"{t_icon} {z['id']} ({z['type']})")

    def eliminar_zona_seleccionada(self):
        sel = self.lst_zonas.curselection()
        if not sel:
            return
        idx = sel[0]
        z_del = self.zonas.pop(idx)
        self.actualizar_lista_box_zonas()
        self.redibujar_canvas()
        self.log(f"🗑️ Zona eliminada: {z_del['id']}")

    def limpiar_todas_las_zonas(self):
        if messagebox.askyesno("Confirmar", "¿Desea eliminar todas las zonas dibujadas?"):
            self.zonas = []
            self.poligono_actual = []
            self.contador_zonas = {"flujo": 0, "interaccion": 0, "exclusion": 0}
            self.actualizar_lista_box_zonas()
            self.redibujar_canvas()

    def guardar_json_zonas(self):
        if not self.zonas:
            messagebox.showwarning("Atención", "No hay zonas creadas para guardar.")
            return None

        vpath = self.video_path.get()
        out_dir = self.carpeta_salida.get().strip() or "resultados"
        os.makedirs(out_dir, exist_ok=True)
        cam_id = self.camera_id.get().strip() or "cam_01"

        contenido = {"camera_id": cam_id, "zones": self.zonas}
        ruta_principal = None

        # 1. Guardar al mismo nivel del video con el mismo nombre del video
        if vpath and os.path.exists(vpath):
            video_dir = os.path.dirname(os.path.abspath(vpath))
            base_name = os.path.splitext(os.path.basename(vpath))[0]
            ruta_principal = os.path.join(video_dir, f"{base_name}_zonas.json")
            with open(ruta_principal, "w", encoding="utf-8") as f:
                json.dump(contenido, f, indent=2)
            self.log(f"💾 File guardado junto al video: {ruta_principal} ({len(self.zonas)} zonas)")

        # 2. Guardar también en la carpeta de salida para compatibilidad con la pipeline
        ruta_salida = os.path.join(out_dir, f"zonas_{cam_id}.json")
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(contenido, f, indent=2)
        self.log(f"💾 File guardado en carpeta de salida: {ruta_salida}")

        messagebox.showinfo("Guardado Exitoso", f"JSON de zonas guardado en:\n{ruta_principal or ruta_salida}")
        return ruta_principal or ruta_salida

    def buscar_y_cargar_json_zonas(self):
        vpath = self.video_path.get()
        initial_dir = os.path.dirname(os.path.abspath(vpath)) if vpath and os.path.exists(vpath) else os.getcwd()
        ruta = filedialog.askopenfilename(
            title="Seleccionar JSON de Zonas Guardadas",
            initialdir=initial_dir,
            filetypes=[("Archivos JSON", "*.json")]
        )
        if ruta:
            self.cargar_json_existente(ruta)

    def cargar_json_existente(self, ruta_json):
        try:
            with open(ruta_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.zonas = data.get("zones", data.get("zonas", []))
            self.contador_zonas = {"flujo": 0, "interaccion": 0, "exclusion": 0}
            for z in self.zonas:
                t = z.get("type", "flujo")
                if t in self.contador_zonas:
                    self.contador_zonas[t] += 1
            self.actualizar_lista_box_zonas()
            self.redibujar_canvas()
            self.log(f"ℹ️ Zonas cargadas desde JSON existente: {os.path.basename(ruta_json)} ({len(self.zonas)} zonas)")
        except Exception as e:
            self.log(f"Error al cargar JSON existente: {e}")

    # -------------------------------------------------------------------------
    # FUNCIONES PESTAÑA 2 (CALIBRACIÓN GUIADA LADO A LADO POR ZONAS JSON)
    # -------------------------------------------------------------------------
    def buscar_json_hom_ref(self):
        ruta = filedialog.askopenfilename(title="Seleccionar JSON de Zonas Cámara Referencia", filetypes=[("Archivos JSON", "*.json")])
        if ruta:
            self.json_ref_path.set(ruta)
            try:
                with open(ruta, "r", encoding="utf-8") as f:
                    self.zonas_hom_ref = json.load(f).get("zones", [])
                self.redibujar_canvas_hom_ref()
                self.log(f"🏷️ JSON Zonas Referencia cargado: {os.path.basename(ruta)} ({len(self.zonas_hom_ref)} zonas)")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def buscar_json_hom_sec(self):
        ruta = filedialog.askopenfilename(title="Seleccionar JSON de Zonas Cámara Secundaria", filetypes=[("Archivos JSON", "*.json")])
        if ruta:
            self.json_sec_path.set(ruta)
            try:
                with open(ruta, "r", encoding="utf-8") as f:
                    self.zonas_hom_sec = json.load(f).get("zones", [])
                self.redibujar_canvas_hom_sec()
                self.log(f"🏷️ JSON Zonas Secundaria cargado: {os.path.basename(ruta)} ({len(self.zonas_hom_sec)} zonas)")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def actualizar_banner_instrucciones_hom(self):
        n_ref = len(self.puntos_hom_ref)
        n_sec = len(self.puntos_hom_sec)

        if n_ref == n_sec:
            if n_ref >= 4:
                txt = f"✅ ¡{n_ref} PUNTOS COMPLETADOS! Presiona '📐 GENERAR MAPAS EN homografia/'"
                col = self.colors["success"]
            else:
                idx = n_ref
                nombre_pt = fusion_mod.NOMBRES_PUNTOS[idx % len(fusion_mod.NOMBRES_PUNTOS)]
                txt = f"👉 PASO {idx+1}/4: Haz clic en {nombre_pt} en la Cámara de Referencia (IZQUIERDA)"
                col = self.colors["warning"]
                self.turno_homografía = "ref"
        else:
            idx = n_sec
            nombre_pt = fusion_mod.NOMBRES_PUNTOS[idx % len(fusion_mod.NOMBRES_PUNTOS)]
            txt = f"👉 PASO {idx+1}/4: AHORA haz clic en el MISMO OBJETO FÍSICO {nombre_pt} en la Cámara Secundaria (DERECHA)"
            col = "#f59e0b"
            self.turno_homografía = "sec"

        self.lbl_instrucciones_hom.configure(text=txt, foreground=col)

    def buscar_parquet_ref(self):
        ruta = filedialog.askopenfilename(title="Seleccionar Parquet Cámara Referencia", filetypes=[("Archivos Parquet", "*.parquet")])
        if ruta:
            self.parquet_ref_path.set(ruta)
            self.log(f"📊 Parquet Referencia seleccionado: {os.path.basename(ruta)}")

    def buscar_parquet_sec(self):
        ruta = filedialog.askopenfilename(title="Seleccionar Parquet Cámara Secundaria", filetypes=[("Archivos Parquet", "*.parquet")])
        if ruta:
            self.parquet_sec_path.set(ruta)
            self.log(f"📊 Parquet Secundario seleccionado: {os.path.basename(ruta)}")

    def cargar_video_hom_ref(self):
        ruta = filedialog.askopenfilename(title="Seleccionar Video de Cámara Referencia", filetypes=[("Videos", "*.mp4 *.avi *.mkv *.mov")])
        if not ruta:
            return

        cap = cv2.VideoCapture(ruta)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.cam_ref_video.set(ruta)
            self.frame_ref = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.h_ref, self.w_ref = self.frame_ref.shape[:2]
            self.redibujar_canvas_hom_ref()
            print(f"\n🌐 [VIDEO REFERENCIA SELECCIONADO]: {os.path.basename(ruta)} | {fps:.2f} FPS | Total frames: {total_frames}\n")
            self.log(f"🌐 Video Referencia cargado: {os.path.basename(ruta)} | {fps:.2f} FPS ({self.w_ref}x{self.h_ref} px)")

    def cargar_video_hom_sec(self):
        ruta = filedialog.askopenfilename(title="Seleccionar Video de Cámara Secundaria", filetypes=[("Videos", "*.mp4 *.avi *.mkv *.mov")])
        if not ruta:
            return

        cap = cv2.VideoCapture(ruta)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.cam_sec_video.set(ruta)
            self.frame_sec = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.h_sec, self.w_sec = self.frame_sec.shape[:2]
            self.redibujar_canvas_hom_sec()
            print(f"\n🌐 [VIDEO SECUNDARIO SELECCIONADO]: {os.path.basename(ruta)} | {fps:.2f} FPS | Total frames: {total_frames}\n")
            self.log(f"🌐 Video Secundario cargado: {os.path.basename(ruta)} | {fps:.2f} FPS ({self.w_sec}x{self.h_sec} px)")

    def redibujar_canvas_hom_ref(self):
        if self.frame_ref is None:
            return

        c_w = self.canvas_hom_ref.winfo_width()
        c_h = self.canvas_hom_ref.winfo_height()
        if c_w < 50 or c_h < 50:
            c_w, c_h = 600, 340

        self.scale_ref_x = c_w / self.w_ref
        self.scale_ref_y = c_h / self.h_ref

        img_resized = cv2.resize(self.frame_ref, (c_w, c_h))
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

        # Dibujar polígonos de zonas si están cargados en la Pestaña 2
        if self.zonas_hom_ref:
            pares_ref_dict = {r: s for s, r in self.emparejamiento_zonas} if self.emparejamiento_zonas else {}
            for z in self.zonas_hom_ref:
                zid = z.get("id", "")
                es_compartida = (zid in pares_ref_dict) or (not self.emparejamiento_zonas)

                if es_compartida:
                    partner = pares_ref_dict.get(zid, zid)
                    color = (0, 255, 200) # Cyan/Verde brillante para góndola compartida
                    grosor = 3
                    txt_tag = f"[COMPARTIDA: {zid} <-> {partner}]"
                else:
                    color = (100, 100, 140) # Gris tenue para góndola no visible en la otra cámara
                    grosor = 1
                    txt_tag = f"[SOLO CAM REF: {zid}]"

                pts = np.array([(int(px * self.scale_ref_x), int(py * self.scale_ref_y)) for px, py in z.get("polygon", [])], np.int32)
                if len(pts) >= 3:
                    cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=grosor)
                    cv2.putText(img_bgr, txt_tag, (pts[0][0] + 5, max(pts[0][1] - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1 if not es_compartida else 2)

        if self.puntos_hom_ref:
            pts_res = [(px * self.scale_ref_x, py * self.scale_ref_y) for px, py in self.puntos_hom_ref]
            img_bgr = fusion_mod.dibujar_puntos_color(img_bgr, pts_res, resaltar_ultimo=True)

        img_resized = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.photo_hom_ref = ImageTk.PhotoImage(Image.fromarray(img_resized))
        self.canvas_hom_ref.delete("all")
        self.canvas_hom_ref.create_image(0, 0, image=self.photo_hom_ref, anchor=tk.NW)

    def redibujar_canvas_hom_sec(self):
        if self.frame_sec is None:
            return

        c_w = self.canvas_hom_sec.winfo_width()
        c_h = self.canvas_hom_sec.winfo_height()
        if c_w < 50 or c_h < 50:
            c_w, c_h = 600, 340

        self.scale_sec_x = c_w / self.w_sec
        self.scale_sec_y = c_h / self.h_sec

        img_resized = cv2.resize(self.frame_sec, (c_w, c_h))
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

        # Dibujar polígonos de zonas si están cargados en la Pestaña 2
        if self.zonas_hom_sec:
            pares_sec_dict = {s: r for s, r in self.emparejamiento_zonas} if self.emparejamiento_zonas else {}
            for z in self.zonas_hom_sec:
                zid = z.get("id", "")
                es_compartida = (zid in pares_sec_dict) or (not self.emparejamiento_zonas)

                if es_compartida:
                    partner = pares_sec_dict.get(zid, zid)
                    color = (0, 255, 200) # Cyan/Verde brillante para góndola compartida
                    grosor = 3
                    txt_tag = f"[COMPARTIDA: {zid} <-> {partner}]"
                else:
                    color = (100, 100, 140) # Gris tenue para góndola no visible en la otra cámara
                    grosor = 1
                    txt_tag = f"[SOLO CAM SEC: {zid}]"

                pts = np.array([(int(px * self.scale_sec_x), int(py * self.scale_sec_y)) for px, py in z.get("polygon", [])], np.int32)
                if len(pts) >= 3:
                    cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=grosor)
                    cv2.putText(img_bgr, txt_tag, (pts[0][0] + 5, max(pts[0][1] - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1 if not es_compartida else 2)

        if self.puntos_hom_sec:
            pts_res = [(px * self.scale_sec_x, py * self.scale_sec_y) for px, py in self.puntos_hom_sec]
            img_bgr = fusion_mod.dibujar_puntos_color(img_bgr, pts_res, resaltar_ultimo=True)

        img_resized = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.photo_hom_sec = ImageTk.PhotoImage(Image.fromarray(img_resized))
        self.canvas_hom_sec.delete("all")
        self.canvas_hom_sec.create_image(0, 0, image=self.photo_hom_sec, anchor=tk.NW)

    def on_canvas_ref_click(self, event):
        if self.frame_ref is None:
            return

        if len(self.puntos_hom_ref) > len(self.puntos_hom_sec):
            messagebox.showinfo("Turno de Marcación", "Por favor marca primero el punto correspondiente en la Cámara Secundaria (Derecha).")
            return

        x_orig = int(event.x / self.scale_ref_x)
        y_orig = int(event.y / self.scale_ref_y)
        x_orig = max(0, min(self.w_ref - 1, x_orig))
        y_orig = max(0, min(self.h_ref - 1, y_orig))

        self.puntos_hom_ref.append([x_orig, y_orig])
        idx = len(self.puntos_hom_ref)
        lbl_pt = fusion_mod.NOMBRES_PUNTOS[(idx - 1) % len(fusion_mod.NOMBRES_PUNTOS)]
        self.log(f"📍 Punto {lbl_pt} marcado en Cámara Referencia (Izquierda): [{x_orig}, {y_orig}]")
        self.redibujar_canvas_hom_ref()
        self.actualizar_banner_instrucciones_hom()

    def on_canvas_sec_click(self, event):
        if self.frame_sec is None:
            return

        if len(self.puntos_hom_sec) >= len(self.puntos_hom_ref):
            messagebox.showinfo("Turno de Marcación", "Por favor marca primero el siguiente punto en la Cámara de Referencia (Izquierda).")
            return

        x_orig = int(event.x / self.scale_sec_x)
        y_orig = int(event.y / self.scale_sec_y)
        x_orig = max(0, min(self.w_sec - 1, x_orig))
        y_orig = max(0, min(self.h_sec - 1, y_orig))

        self.puntos_hom_sec.append([x_orig, y_orig])
        idx = len(self.puntos_hom_sec)
        lbl_pt = fusion_mod.NOMBRES_PUNTOS[(idx - 1) % len(fusion_mod.NOMBRES_PUNTOS)]
        self.log(f"📍 Punto {lbl_pt} marcado en Cámara Secundaria (Derecha): [{x_orig}, {y_orig}]")
        self.redibujar_canvas_hom_sec()
        self.actualizar_banner_instrucciones_hom()

    def deshacer_ultimo_punto_homografia(self):
        if len(self.puntos_hom_ref) > len(self.puntos_hom_sec):
            self.puntos_hom_ref.pop()
        elif len(self.puntos_hom_sec) > 0:
            self.puntos_hom_sec.pop()
        self.redibujar_canvas_hom_ref()
        self.redibujar_canvas_hom_sec()
        self.actualizar_banner_instrucciones_hom()
        self.log("↩️ Deshecho el último clic de calibración.")

    def limpiar_puntos_homografia_lado_a_lado(self):
        self.puntos_hom_ref = []
        self.puntos_hom_sec = []
        self.redibujar_canvas_hom_ref()
        self.redibujar_canvas_hom_sec()
        self.actualizar_banner_instrucciones_hom()
        self.log("⚠️ Puntos de homografía borrados en ambas cámaras.")

    def resolver_homografia_y_guardar_carpeta_homografia(self):
        cid_ref = self.cam_ref_id.get().strip() or "cam_04"
        cid_sec = self.cam_sec_id.get().strip() or "cam_05"

        if len(self.puntos_hom_ref) < 4 or len(self.puntos_hom_sec) < 4:
            messagebox.showwarning(
                "Puntos Insuficientes",
                f"Se requieren al menos 4 puntos correspondientes marcados P1..P4 tanto en la cámara de referencia '{cid_ref}' "
                f"como en la cámara secundaria '{cid_sec}'.\n\n"
                f"Llevas:\n• Ref ({cid_ref}): {len(self.puntos_hom_ref)} puntos\n• Sec ({cid_sec}): {len(self.puntos_hom_sec)} puntos"
            )
            return

        if self.frame_ref is None:
            messagebox.showerror("Error", "Cargue primero el video de la Cámara de Referencia.")
            return

        carpeta_hom = os.path.join(dir_actual, "homografia")
        os.makedirs(carpeta_hom, exist_ok=True)

        try:
            H, error_medio = fusion_mod.calcular_homografia(self.puntos_hom_sec, self.puntos_hom_ref)
            self.homografias[cid_sec] = H.tolist()
            self.homografias[cid_ref] = np.eye(3).tolist()

            if error_medio > 45.0:
                messagebox.showwarning(
                    "⚠️ Posible Desalineación de Puntos",
                    f"El error de reproyección medio es de {error_medio:.1f} píxeles.\n\n"
                    f"Asegúrese de que P1(Morado), P2(Verde), P3(Naranja) y P4(Amarillo) correspondan exactamente al MISMO objeto o esquina física en el plano de ambas cámaras."
                )

            matriz_json = os.path.join(carpeta_hom, "homografia_matriz.json")
            with open(matriz_json, "w", encoding="utf-8") as f:
                json.dump({"ref_cam": cid_ref, "sec_cam": cid_sec, "H": H.tolist(), "reprojection_error_px": error_medio}, f, indent=2)

            pq_ref = self.parquet_ref_path.get().strip()
            pq_sec = self.parquet_sec_path.get().strip()

            if not pq_ref or not os.path.exists(pq_ref):
                candidatos = glob.glob(f"datos_parquet/tracking_{cid_ref}.parquet") + glob.glob(f"salida_*/tracking_{cid_ref}.parquet") + glob.glob("*.parquet")
                if candidatos:
                    pq_ref = candidatos[0]

            if not pq_sec or not os.path.exists(pq_sec):
                candidatos = glob.glob(f"datos_parquet/tracking_{cid_sec}.parquet") + glob.glob(f"salida_*/tracking_{cid_sec}.parquet") + glob.glob("*.parquet")
                if candidatos:
                    pq_sec = candidatos[0]

            dfs_dict = {}
            if pq_ref and os.path.exists(pq_ref):
                dfs_dict[cid_ref] = pd.read_parquet(pq_ref)
                self.log(f"  • Parquet Referencia cargado: {pq_ref}")

            if pq_sec and os.path.exists(pq_sec):
                dfs_dict[cid_sec] = pd.read_parquet(pq_sec)
                self.log(f"  • Parquet Secundario cargado: {pq_sec}")

            if not dfs_dict:
                messagebox.showwarning(
                    "Parquets no Encontrados",
                    "No se encontraron los archivos .parquet de tracking para las cámaras.\n"
                    "Por favor, seleccione manualmente los archivos .parquet con los botones '📊 Parquet Ref' y '📊 Parquet Sec'."
                )
                return

            fondo_ref = cv2.cvtColor(self.frame_ref, cv2.COLOR_RGB2BGR)

            salida_mapa = fusion_mod.generar_mapa_calor_multicamara_fusionado(
                dfs_dict=dfs_dict,
                homografias_dict=self.homografias,
                cam_ref_id=cid_ref,
                fondo_ref=fondo_ref,
                carpeta_salida=carpeta_hom
            )

            if self.frame_sec is not None:
                fondo_sec = cv2.cvtColor(self.frame_sec, cv2.COLOR_RGB2BGR)
                fusion_mod.generar_vista_2d_superpuesta_warped(
                    frame_ref=fondo_ref,
                    frame_sec=fondo_sec,
                    H=H,
                    carpeta_salida=carpeta_hom
                )

            ruta_abs_hom = os.path.abspath(carpeta_hom)
            self.log("\n" + "=" * 60)
            self.log(f"🎉 HOMOGRAFÍA Y COMPROBACIONS VISUALES COMPLETADAS CON ÉXITO")
            self.log(f"📏 Error Medio de Reproyección: {error_medio:.2f} px")
            self.log(f"📁 Entregables y Diagnósticos 2D guardados en: {ruta_abs_hom}")
            self.log("=" * 60)

            messagebox.showinfo(
                "¡Fusión de Homografía Exitosa!",
                f"Se ha calculado la homografía (Error: {error_medio:.1f}px) y generado los mapas y diagnósticos visuales 2D.\n\n"
                f"📁 Carpeta de Salida:\n{ruta_abs_hom}\n\n"
                f"Imágenes de Verificación Generadas:\n"
                f"1. mapa_calor_planta_2d_topdown.png (Heatmap 2D Cenital libre de deformación)\n"
                f"2. verificacion_trayectorias_planta_2d.png (Pies/Trayectorias 2D Cenital)\n"
                f"3. mapa_calor_multicamara_fusionado.png (Heatmap Unificado sobre Ref)\n"
                f"4. verificacion_superposicion_optica_2d.png (Superposición 50/50 de perspectivas)\n\n"
                f"Archivos de Datos:\n"
                f"• tracking_multicamara_unificado.parquet\n"
                f"• homografia_matriz.json"
            )

        except Exception as e:
            messagebox.showerror("Error de Homografía", str(e))

    # -------------------------------------------------------------------------
    # EJECUCIÓN MULTIHILO DEL PIPELINE DE EXTRACCIÓN (PASO 1)
    # -------------------------------------------------------------------------
    def iniciar_pipeline(self):
        vpath = self.video_path.get()
        if not vpath or not os.path.exists(vpath):
            messagebox.showerror("Error", "Seleccione un video válido antes de iniciar.")
            return

        out_dir = self.carpeta_salida.get().strip() or "resultados"
        os.makedirs(out_dir, exist_ok=True)

        cam_id = self.camera_id.get().strip() or "cam_01"
        salida_parquet = os.path.join(out_dir, f"tracking_{cam_id}.parquet")

        # Guardar automáticamente el JSON de zonas si existen zonas dibujadas
        if self.zonas:
            self.guardar_json_zonas()

        self.tracking_running = True
        self.btn_run_tracking.configure(state="disabled")
        self.lbl_status.configure(text="⏳ Procesando Tracking YOLO Pose (En segundo plano)...", foreground=self.colors["warning"])
        self.log(f"🚀 Iniciando extracción de datos para cámara [{cam_id}] en segundo plano...")

        t = threading.Thread(target=self.worker_tracking, args=(vpath, cam_id, salida_parquet, out_dir), daemon=True)
        t.start()

    def worker_tracking(self, video_path, camera_id, salida_parquet, carpeta_salida):
        class QueueLogger:
            def __init__(self, queue):
                self.queue = queue
            def write(self, msg):
                if msg and msg.strip():
                    self.queue.put(("log", msg.strip()))
            def flush(self):
                pass

        orig_stdout = sys.stdout
        sys.stdout = QueueLogger(self.log_queue)
        try:
            disp_val = "cuda" if "GPU" in self.dispositivo_opt.get() else "cpu"
            extraer_mod.modo_guardar(
                video_path=video_path,
                camera_id=camera_id,
                modelo_path=self.obtener_ruta_modelo_absoluta(),
                imgsz=self.imgsz.get(),
                salida_parquet=salida_parquet,
                carpeta_salida=carpeta_salida,
                frame_skip=self.frame_skip.get(),
                conf=self.conf_thresh.get(),
                dispositivo=disp_val
            )
            self.log_queue.put(("done", salida_parquet))
        except Exception as e:
            self.log_queue.put(("error", str(e)))
        finally:
            sys.stdout = orig_stdout

    def procesar_mensajes_queue(self):
        while not self.log_queue.empty():
            msg_type, content = self.log_queue.get()
            if msg_type == "log":
                self.log(content)
                if "%" in content or "Frame" in content:
                    self.actualizar_progreso_por_texto(content)

            elif msg_type == "done":
                self.tracking_running = False
                self.tracking_complete = True
                self.btn_run_tracking.configure(state="normal")
                self.lbl_status.configure(text="✅ Extracción completada. Procesando analítica...", foreground=self.colors["success"])
                self.progress_bar["value"] = 100
                self.log(f"🎉 Tracking finalizado. Parquet guardado en: {content}")
                self.log("🚀 Iniciando automático de procesado de zonas, mapas y métricas...")
                self.root.after(500, self.generar_mapas_y_metricas)

            elif msg_type == "error":
                self.tracking_running = False
                self.btn_run_tracking.configure(state="normal")
                self.lbl_status.configure(text="❌ Error en la inferencia", foreground=self.colors["danger"])
                self.log(f"❌ Error en tracking: {content}")
                messagebox.showerror("Error de Inferencia", content)

        self.root.after(100, self.procesar_mensajes_queue)

    def actualizar_progreso_por_texto(self, texto):
        try:
            if "(" in texto and "%)" in texto:
                pct_str = texto.split("(")[1].split("%")[0]
                pct = float(pct_str)
                self.progress_bar["value"] = pct
                self.lbl_status.configure(text=f"⏳ Procesando: {pct:.1f}% completo...")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # GENERACIÓN DE MAPAS Y MÉTRICAS (PASO 3 - UNIFICADO)
    # -------------------------------------------------------------------------
    def generar_mapas_y_metricas(self):
        out_dir = self.carpeta_salida.get().strip() or "resultados"
        cam_id = self.camera_id.get().strip() or "cam_01"
        salida_parquet = os.path.join(out_dir, f"tracking_{cam_id}.parquet")
        video_path = self.video_path.get()

        if not os.path.exists(salida_parquet):
            messagebox.showerror("Error", f"No se encontró el Parquet de tracking en:\n{salida_parquet}\n\nEjecute la extracción primero.")
            return

        ruta_json = os.path.join(out_dir, f"zonas_{cam_id}.json")
        if not self.zonas:
            if os.path.exists(ruta_json):
                self.cargar_json_existente(ruta_json)
            elif os.path.exists(f"zonas_{cam_id}.json"):
                self.cargar_json_existente(f"zonas_{cam_id}.json")

        if self.zonas:
            self.guardar_json_zonas()

        self.log("\n" + "=" * 60)
        self.log("🚀 INICIANDO POST-PROCESAMIENTO: MAPAS DE CALOR Y ANALÍTICA RETAIL")
        self.log("=" * 60)
        self.lbl_status.configure(text="⏳ Generando mapas y analítica...", foreground=self.colors["warning"])

        def worker_post_procesamiento():
            try:
                self.log("  [1/3] Generando suite de 6 mapas de calor...")
                mapas_mod.main(
                    parquet_path=salida_parquet,
                    video_path=video_path,
                    zonas_path=ruta_json if os.path.exists(ruta_json) else None,
                    carpeta_salida=out_dir
                )

                self.log("  [2/3] Clasificando Pick-Up vs Put-Back y exportando reportes...")
                procesar_mod.main(
                    parquet_path=salida_parquet,
                    zonas_json_path=ruta_json if os.path.exists(ruta_json) else None,
                    video_path=video_path,
                    carpeta_salida=out_dir
                )

                if self.generar_video_anotado.get():
                    self.log("  [3/3] Renderizando video anotado con pose y métricas...")
                    base_name = os.path.splitext(os.path.basename(video_path))[0]
                    nombre_video_salida = f"video_anotado_{base_name}.mp4"

                    anotar_mod.anotar_video(
                        video_path=video_path,
                        parquet_path=salida_parquet,
                        zonas_path=ruta_json if os.path.exists(ruta_json) else None,
                        carpeta_salida=out_dir,
                        salida_filename=nombre_video_salida
                    )
                else:
                    self.log("  [3/3] ⏩ Omisión de video anotado por configuración del Checkbox.")

                self.root.after(0, lambda d=out_dir: self.post_procesamiento_completado(d))
            except Exception as e:
                err_msg = str(e)
                self.log(f"❌ Error en post-procesamiento: {err_msg}")
                self.root.after(0, lambda msg=err_msg: messagebox.showerror("Error de Procesamiento", msg))

        t_post = threading.Thread(target=worker_post_procesamiento, daemon=True)
        t_post.start()

    def post_procesamiento_completado(self, out_dir):
        self.log("\n" + "=" * 60)
        self.log(f"✅ PROCESO COMPLETADO EXITOSAMENTE")
        self.log(f"📁 Todos los entregables guardados en: {os.path.abspath(out_dir)}")
        self.log("=" * 60)

        archivos_generados = glob.glob(os.path.join(out_dir, "*"))
        msj_archivos = "\n".join([f"• {os.path.basename(a)}" for a in archivos_generados])

        messagebox.showinfo("¡Proceso Finalizado!", f"Se han generado todos los reportes y mapas en la carpeta única:\n\n{out_dir}\n\nContenido:\n{msj_archivos}")

    def log(self, mensaje):
        self.txt_log.insert(tk.END, mensaje + "\n")
        self.txt_log.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = RetailTrackerApp(root)
    root.mainloop()
