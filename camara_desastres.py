import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar modelo entrenado y nombres de clases (en el mismo orden que entrenó)
modelo = load_model("modelo_desastres.h5")
clases = ["Incendio", "Inundacion", "Terremoto", "Tornado"]

# 0 abre la cámara web (o pon la ruta de un archivo: 'video_desastre.mp4')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Pasar a RGB, redimensionar a 48x48 y normalizar (0 a 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (48, 48)) / 255.0

    # 2. Predecir el desastre natural en el fotograma
    probs = modelo(np.array([img], dtype=np.float32), training=False).numpy()[0]
    conf = float(np.max(probs))
    nombre = clases[int(np.argmax(probs))]

    # 3. Solo mostrar desastre si la confianza supera el 75%
    if conf >= 0.75:
        texto = f"Desastre: {nombre} ({conf*100:.1f}%)"
        color = (0, 255, 0)
    else:
        texto = f"Sin desastre claro ({nombre} {conf*100:.0f}%)"
        color = (0, 255, 255)

    cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Detector de Desastres - Camara", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
