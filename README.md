# uCCA (unsupervised Centroid Clustering Algorithm)

**uCCA (unsupervised Centroid Clustering Algorithm)**  is a project that combines image processing techniques and centroid analysis to study circular patterns in images in a semi-supervised manner. The project consists of two main components:

1. **Processing**: A graphical user interface (GUI) developed in Python that allows users to load images, select regions of interest (ROI), and automatically crop circles detected in those images. Subsequently, it obtains the centroids of the region of interest.
2. **Circles**: An algorithm that calculates the difference between theoretical (expected) centroids and experimental centroids (obtained from the detected circles).
---
## Abstract

We apply new Machine Learning (ML) technologies to optimize the Bi-Ronchi and Hartmann tests (BRT and HT, respectively), regarding the recognition, identification, and location of the centroids in experimental Bi-Ronchigrams and Hartmanngrams. We replace the conventional rigid Hartmann screen with structured apertures implemented via a Spatial Light Modulator (SLM) which enables the generation of multiple patterns with different aperture geometries. In the case of the Bi-Ronchi Mask (BRM) the geometry consists of square apertures in the BRT, whereas the Hartmann mask (HM) uses circular apertures in the HT. We designed an experimental setup based on an SLM with a laser illumination system and implemented an unsupervised Centroid Neural Network (uCNN), based on the Machine Learning algorithm $K-$Means, to identify the geometries of the centroids followed by their segmentation and localization by clustering. We compare the experimental and theoretical Bi-Ronchigrams (or Hartmanngrams) to obtain a Point Cloud of transverse aberrations ($TA$), denoted as $PC_{TA}$. We apply the Point Cloud Method (PCM) to obtain an integrable surface from the points in $PC_{TA}$. Finally, we replace the numerical integration of ${PC_{TA}} {\to} {TA}$ with a directional derivative approach based on the Eikonal equation, solved using gaussian quadrature to obtain the wavefront. We compare our results with the Zernike aberration polynomials for sensing optical elements from the aberrations of the system by means of the aberrations of its Wavefront ${\mathcal{W}} ({\rho}, {\theta})$.

## Resumen

Se aplicamron nuevas tecnologías de Aprendizaje Automático (Machine Learning, ML) para optimizar las pruebas de Bi-Ronchi y Hartmann (BRT y HT, respectivamente), en lo que respecta al reconocimiento, identificación y localización de los centroides en Bi-Ronchigramas y Hartmanngramas experimentales. Reemplazamos la pantalla rígida convencional de Hartmann por aperturas estructuradas implementadas mediante un Modulador Espacial de Luz (SLM), lo que permite la generación de múltiples patrones con diferentes geometrías de apertura. En el caso de la Máscara Bi-Ronchi (BRM), la geometría consiste en aperturas cuadradas en la BRT, mientras que la máscara de Hartmann (HM) utiliza aperturas circulares en la HT.

Diseñamos un montaje experimental basado en un SLM con un sistema de iluminación láser e implementamos una Red Neuronal de Centroides no Supervisada (uCNN), basada en el algoritmo de Aprendizaje Automático K-Means, para identificar las geometrías de los centroides, seguida de su segmentación y localización mediante agrupamiento (clustering). Comparamos los Bi-Ronchigramas (o Hartmanngramas) experimentales y teóricos para obtener una Nube de Puntos de Aberraciones Transversales ($TA$), denotada como $PC_{TA}$. Aplicamos el Método de Nube de Puntos (PCM) para obtener una superficie integrable a partir de los puntos en $PC_{TA}$.

Finalmente, reemplazamos la integración numérica de ${PC_{TA}} \to {TA}$ por un enfoque de derivadas direccionales basado en la ecuación de Eikonal, resuelta mediante cuadratura gaussiana para obtener el frente de onda. Comparamos nuestros resultados con los polinomios de aberración de Zernike para analizar elementos ópticos a partir de las aberraciones del sistema mediante las aberraciones de su Frente de Onda ${\mathcal{W}} ({\rho}, {\theta})$.

## Features
- **Intuitive graphical interface**: Load and crop circles of interest in images with a Tkinter-based interface.
- **ROI selection**: Allows defining movable regions of interest with the mouse for custom cropping.
- **Automatic detection**: Identifies centroids in images using processing techniques such as CLAHE, bilateral filtering, and contour analysis.
- **Centroid analysis**: Compares theoretical and experimental centroids to evaluate deviations.
- **Flexible saving**: Exports cropped images and results in formats like JPG and JSON.

---

## Requeriments
- Python 3.12
- Libraries:
  - `opencv-python` 
  - `tkinter`
  - `Pillow` 
  - `numpy`
  - `os`
  - `numpy`
  - `matplotlib`
  - `csv`
  - `json`


