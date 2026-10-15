import cv2
from ultralytics import YOLO

# 1. Cargar el modelo
model = YOLO("best.pt")

# Ver qué clases tiene tu modelo registradas
print("\n" + "="*40)
print("CLASES QUE EL MODELO RECONOCE:")
for idx, name in model.names.items():
    print(f"  Clase {idx}: {name}")
print("="*40 + "\n")

# 2. Abrir cámara en resolución fluida para CPU (640x480)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

LIMITE_TUERCAS = 20

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # imgsz=320 hace que la CPU vuele rápido
    # conf=0.10 detecta incluso si la seguridad es muy baja
    results = model(frame, imgsz=320, conf=0.10, verbose=False)

    total_tuercas = 0

    for box in results[0].boxes:
        total_tuercas += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confianza = float(box.conf[0])
        clase_id = int(box.cls[0])
        nombre_clase = model.names[clase_id]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Dibujar
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(frame, f"{nombre_clase} {confianza:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    excedente = max(0, total_tuercas - LIMITE_TUERCAS)

    cv2.putText(frame, f"Detectadas: {total_tuercas}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Excedente: {excedente}", (15, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if excedente > 0 else (0, 255, 0), 2)

    cv2.imshow("Prueba Dobot", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()