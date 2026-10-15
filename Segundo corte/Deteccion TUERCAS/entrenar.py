from ultralytics import YOLO

def main():
    # 1. Cargamos el modelo base ligero
    model = YOLO("yolov8n.pt")

    # 2. Entrenamos con tu dataset
    print("Iniciando entrenamiento del modelo...")
    model.train(
        data=r"D:\Adri564\Documents\tuercas\Data.yaml", 
        epochs=30,      # 30 pasadas
        imgsz=640,      # Resolución estándar
        batch=16,       # Si tu PC se traba o no tiene mucha RAM, cambia a 8
        device="cpu"    # Si tienes gráfica NVIDIA con CUDA puedes cambiarlo a 0
    )

    print("¡Entrenamiento finalizado exitosamente!")

if __name__ == "__main__":
    main()