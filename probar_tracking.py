from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")  # se descarga solo la primera vez (necesita internet)

results = model.track(
    source="clip_test.mp4",
    tracker="bytetrack.yaml",
    save=True,
    persist=True,
    classes=[0],  # clase 0 = persona
    conf=0.4
)

# Verificar cuántos IDs únicos detectó
ids_vistos = set()
for r in results:
    if r.boxes.id is not None:
        ids_vistos.update(r.boxes.id.int().tolist())
print("IDs únicos detectados:", len(ids_vistos))