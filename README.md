# UnsupervicedNNBiRonHart-CuNN

**UnsupervicedNNBiRonHart-CuNN**  is a project that combines image processing techniques and centroid analysis to study circular patterns in images in a semi-supervised manner. The project consists of two main components:

1. **Processing**: A graphical user interface (GUI) developed in Python that allows users to load images, select regions of interest (ROI), and automatically crop circles detected in those images. Subsequently, it obtains the centroids of the region of interest.
2. **Circles**: An algorithm that calculates the difference between theoretical (expected) centroids and experimental centroids (obtained from the detected circles).
---

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


