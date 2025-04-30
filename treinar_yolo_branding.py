from ultralytics import YOLO

# Caminho do modelo base (pode usar yolov8s.pt para mais precisão)
modelo_base = "yolov8n.pt"

# Caminho do arquivo data.yaml
caminho_yaml = "branding_dataset/data.yaml"

# Carregar modelo base
model = YOLO(modelo_base)

# Iniciar o treino
model.train(
    data=caminho_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    name="detector_pragmatic",
    project="runs/train"
)
