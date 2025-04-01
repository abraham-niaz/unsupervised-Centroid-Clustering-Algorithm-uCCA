# UnsupervicedNNBiRonHart-CuNN

**UnsupervicedNNBiRonHart-CuNN** es un proyecto que combina técnicas de procesamiento de imágenes y análisis de centroides para estudiar patrones circulares en imágenes de manera semi supervisada. El proyecto consta de dos componentes principales:

1. **Processing**: Una interfaz gráfica (GUI) desarrollada en Python que permite a los usuarios cargar imágenes, seleccionar regiones de interés (ROI) y recortar círculos automáticamente detectados en dichas imágenes. Posteriormente obtener los centroides de la región de interés.
2. **Circles**: Un algoritmo que calcula la diferencia entre los centroides teóricos (esperados) y los centroides experimentales (obtenidos de los círculos detectados).

---

## Características
- **Interfaz gráfica intuitiva**: Carga, visualiza y recorta círculos de interés en imágenes con una interfaz basada en Tkinter.
- **Selección de ROI**: Permite definir regiones de interés desplazables con el ratón para un recorte personalizado.
- **Detección automática**: Identifica los centroides en imágenes mediante técnicas de procesamiento como CLAHE, filtrado bilateral y análisis de contornos.
- **Análisis de centroides**: Compara centroides teóricos y experimentales para evaluar desviaciones.
- **Guardado flexible**: Exporta imágenes recortadas y resultados en formatos como JPG y JSON.

---

## Requisitos
- Python 3.12
- Bibliotecas:
  - `opencv-python` 
  - `tkinter`
  - `Pillow` 
  - `numpy`
  - `os`
  - `numpy`
  - `matplotlib`
  - `csv`
  - `json`


