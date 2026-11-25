# Ai_Voy

Proyecto de redes neuronales convolucionales (CNN) para clasificación de desastres naturales y detección/reconocimiento facial.

## Contenido

- `camara_desastres.py`: Detector de desastres naturales en tiempo real usando la cámara web y `modelo_desastres.h5`.
- `CNNriesgo_desastres.ipynb`: Entrenamiento y evaluación del modelo de desastres naturales (`modelo_desastres.h5`).
- `CNN.ipynb` / `CNNriesgo.ipynb`: Entrenamiento y pruebas del modelo de reconocimiento facial (`modelo_caras.h5`).
- `desastres_dataset/`: Dataset de imágenes de desastres.
- `haarcascade_dataset/`: Dataset de rostros y clasificador Haar Cascade.
- `videos_desastres/`: Videos de prueba.
- `requirements.txt`: Dependencias exactas de Python (probado con Python 3.12).

## Instalación en otra PC

1. Clona el repositorio:
   ```bash
   git clone https://github.com/HugoL64/Ai_Voy.git
   cd Ai_Voy
   ```

2. Crea y activa un entorno virtual (Python 3.12 recomendado):
   ```bash
   python3.12 -m venv .venv
   # En Linux/macOS:
   source .venv/bin/activate
   # En Windows (PowerShell):
   # .venv\Scripts\Activate.ps1
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Ejecuta el detector con cámara:
   ```bash
   python camara_desastres.py
   ```
   Presiona `q` para cerrar la ventana.
