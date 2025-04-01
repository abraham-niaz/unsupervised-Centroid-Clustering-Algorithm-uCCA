# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 17:30:19 2025

@author: Abraham
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from procesar import *
from PIL import Image, ImageTk
import os
import json
import numpy as np

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application")
        self.root.geometry("1200x700")

        # Variables
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.scale = 1.0
        self.roi_rect = None  # ID del rectángulo en el canvas
        self.roi_start = None  # Coordenadas iniciales del ROI
        self.roi_coords = None  # Coordenadas actuales del ROI (x1, y1, x2, y2)
        
        # Create GUI elements
        self.create_widgets()
        # Vincular el cambio de modo a una función
        self.mode.trace('w', self.update_area_params)

    def create_widgets(self):
        # Menu Frame
        menu_frame = tk.Frame(self.root)
        menu_frame.pack(pady=10)
        
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_slider = tk.Scale(
            menu_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
            label="Image Scale", variable=self.scale_var, length=200, command=self.update_image_scale
        )
        self.scale_slider.pack(side=tk.LEFT, padx=5)
        
        #self.root.bind("<Left>", self.update_scale_with_keys)
        #self.root.bind("<Right>", self.update_scale_with_keys)

        tk.Button(menu_frame, text="Load image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(menu_frame, text="Process image", command=self.process_image).pack(side=tk.LEFT, padx=5)
        tk.Button(menu_frame, text="Save results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        tk.Button(menu_frame, text="Crop ROI", command=self.crop_roi).pack(side=tk.LEFT, padx=5)  # Botón para recortar ROI
        
        # Mode selection
        mode_frame = tk.LabelFrame(self.root, text="Processing mode")
        mode_frame.pack(pady=5, padx=10, fill="x")
        self.mode = tk.StringVar(value="recortar")
        tk.Radiobutton(mode_frame, text="Cut circles", variable=self.mode, value="recortar").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_frame, text="Circles processing", variable=self.mode, value="procesar").pack(side=tk.LEFT, padx=5)

        # Parameters Frame (solo se muestra el inicio)
        param_frame = tk.LabelFrame(self.root, text="Parameters")
        param_frame.pack(pady=10, padx=10, fill="x")
        tk.Label(param_frame, text="Min area:").grid(row=0, column=5, padx=5, pady=5)
        self.area_min = tk.Scale(param_frame, resolution=5, orient=tk.HORIZONTAL, length=200)#self.process_images
        self.area_min.set(100)
        self.area_min.grid(row=0, column=6, padx=5, pady=5)

        tk.Label(param_frame, text="Max area:").grid(row=1, column=0, padx=5, pady=5)
        self.area_max = tk.Scale(param_frame, resolution=5, orient=tk.HORIZONTAL, length=200)#self.process_images
        self.area_max.set(5000)
        self.area_max.grid(row=1, column=1, padx=5, pady=5)
        
        # Establecer valores iniciales
        self.update_area_params()

        # Threshold parameters
        tk.Label(param_frame, text="Threshold Mode:").grid(row=0, column=0, padx=5, pady=5)
        self.threshold_mode = tk.StringVar(value="otsu")
        tk.Radiobutton(param_frame, text="Otsu", variable=self.threshold_mode, value="otsu").grid(row=0, column=1)
        tk.Radiobutton(param_frame, text="Manual", variable=self.threshold_mode, value="manual").grid(row=0, column=2)

        tk.Label(param_frame, text="Manual Threshold:").grid(row=0, column=3, padx=5, pady=5)
        self.manual_threshold = tk.Scale(param_frame, from_=1, to=255, resolution=5, orient=tk.HORIZONTAL, length=200)
        self.manual_threshold.set(127)
        self.manual_threshold.grid(row=0, column=4, padx=5, pady=5)
        # Kernel parameters
        tk.Label(param_frame, text="Gaussian Kernel:").grid(row=3, column=0, padx=5, pady=5)
        self.gaussian_kernel = tk.Scale(param_frame, from_=1, to=31, resolution=2, orient=tk.HORIZONTAL, length=200)
        self.gaussian_kernel.set(9)
        self.gaussian_kernel.grid(row=3, column=1, padx=5, pady=5)
        
        tk.Label(param_frame, text="Bilateral d:").grid(row=3, column=2, padx=5, pady=5)
        self.bilateral_d = tk.Scale(param_frame, from_=1, to=31, resolution=2, orient=tk.HORIZONTAL, length=200)
        self.bilateral_d.set(7)
        self.bilateral_d.grid(row=3, column=3, padx=5, pady=5)
        
        # Morphology parameters
        tk.Label(param_frame, text="Morphology:").grid(row=4, column=0, padx=5, pady=5)
        self.apply_erosion = tk.BooleanVar()
        tk.Checkbutton(param_frame, text="Erosion", variable=self.apply_erosion).grid(row=4, column=1)
        self.apply_dilation = tk.BooleanVar()
        tk.Checkbutton(param_frame, text="Dilation", variable=self.apply_dilation).grid(row=4, column=2)

        tk.Label(param_frame, text="Kernel Size:").grid(row=4, column=3, padx=5, pady=5)
        self.morph_kernel = tk.Scale(param_frame, from_=1, to=15, resolution=2, orient=tk.HORIZONTAL, length=200)
        self.morph_kernel.set(3)
        self.morph_kernel.grid(row=4, column=4, padx=5, pady=5)

        # CLAHE parameters
        tk.Label(param_frame, text="CLAHE Clip Limit:").grid(row=1, column=3, padx=5, pady=5)
        self.clahe_clip = tk.Scale(param_frame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, length=200)#self.process_image
        self.clahe_clip.set(1.0)
        self.clahe_clip.grid(row=1, column=4, padx=5, pady=5)

        tk.Label(param_frame, text="CLAHE Tile Size:").grid(row=1, column=5, padx=5, pady=5)
        self.clahe_tile = tk.Scale(param_frame, from_=1, to=20, orient=tk.HORIZONTAL, length=200)#self.process_images
        self.clahe_tile.set(4)
        self.clahe_tile.grid(row=1, column=6, padx=5, pady=5)

        # Canvas for image display
        self.canvas = tk.Canvas(self.root, width=1000, height=700)
        self.canvas.pack(pady=10)

        # Vincular eventos del mouse para el ROI
        self.canvas.bind("<Button-1>", self.start_roi)
        self.canvas.bind("<B1-Motion>", self.update_roi)
        self.canvas.bind("<ButtonRelease-1>", self.finalize_roi)

        # Status bar
        self.status = tk.Label(self.root, text="Ok", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        
    def update_area_params(self, *args):
        """Actualiza los parámetros de área (valores y rangos) según el modo seleccionado"""
        current_mode = self.mode.get()
        
        if current_mode == "procesar":
            # Rangos y valores más bajos para el modo "procesar"
            self.area_min.config(from_=0, to=500)    # Ejemplo: tope máximo de 5000
            self.area_min.set(50)                     # Valor inicial más bajo
            self.area_max.config(from_=0, to=5000)   # Ejemplo: tope máximo de 10000
            self.area_max.set(2000)                   # Valor inicial más bajo
        elif current_mode == "recortar":
            # Rangos y valores originales para el modo "recortar"
            self.area_min.config(from_=0, to=150000)  # Tope original
            self.area_min.set(100)                    # Valor inicial original
            self.area_max.config(from_=150000, to=300000)  # Tope original
            self.area_max.set(5000)                   # Valor inicial original

    def update_scale_with_keys(self, event):
        """Ajusta la escala con las teclas izquierda/derecha."""
        step = 0.05  # Incremento o decremento
        current_value = self.scale_var.get()

        if event.keysym == "Left":
            new_value = max(self.scale_slider.cget("from"), current_value - step)
        elif event.keysym == "Right":
            new_value = min(self.scale_slider.cget("to"), current_value + step)

        self.scale_var.set(new_value)  # Actualiza el slider 
        
    def display_image(self, img, scale=1.0):
        """Actualiza la imagen mostrada"""
        display_img = img.copy()
        if len(display_img.shape) == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
        else:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w = display_img.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_resized))
        self.canvas.delete("all")
        self.canvas.create_image(500, 150, image=self.photo)
        self.scale = scale  # Actualizar escala actual

    def update_image_scale(self, value):
        """Actualiza la imagen recortada """
        if self.original_image is not None:
            scale = self.scale_var.get()
            self.display_image(self.original_image, scale)
            if self.roi_coords:  # Redibujar ROI si existe
                self.redraw_roi()

    def load_image(self):
        """Carga la imagen"""
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                messagebox.showerror("Error", f"Error loading image: {self.image_path}")
                self.status.config(text="Error loading image")
                return
            self.display_image(self.original_image)
            self.status.config(text=f"Loaded: {os.path.basename(self.image_path)}")
            self.roi_coords = None  # Reiniciar ROI al cargar nueva imagen
            
    def update_roi(self, event):
        """Actualiza el tamaño del ROI mientras se arrastra el mouse."""
        if self.roi_start:
            self.canvas.coords(self.roi_rect, self.roi_start[0], self.roi_start[1], event.x, event.y)


    def start_roi(self, event):
        """Inicia la selección del ROI al hacer clic."""
        if self.original_image is None:
            return
        self.roi_start = (event.x, event.y)
        if self.roi_rect:
            self.canvas.delete(self.roi_rect)
        self.roi_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="red", width=2
        )
        
    def update_roi(self, event):
        """Actualiza el tamaño del ROI mientras se arrastra el mouse."""
        if self.roi_start:
            self.canvas.coords(self.roi_rect, self.roi_start[0], self.roi_start[1], event.x, event.y)
            
    def finalize_roi(self, event):
        """Finaliza la selección del ROI al soltar el mouse."""
        if self.roi_start:
            self.roi_coords = (
                min(self.roi_start[0], event.x),
                min(self.roi_start[1], event.y),
                max(self.roi_start[0], event.x),
                max(self.roi_start[1], event.y)
            )
            self.roi_start = None
            self.status.config(text=f"ROI selected: {self.roi_coords}")

    def redraw_roi(self):
        """Redibuja el ROI al cambiar la escala de la imagen."""
        if self.roi_coords and self.roi_rect:
            self.canvas.delete(self.roi_rect)
            self.roi_rect = self.canvas.create_rectangle(
                self.roi_coords[0], self.roi_coords[1], self.roi_coords[2], self.roi_coords[3],
                outline="red", width=2
            )

    def crop_roi(self):
        """Recorta la región seleccionada por el ROI y la muestra."""
        if self.original_image is None or self.roi_coords is None:
            messagebox.showerror("Error", "Please load an image and select a ROI first")
            return

        # Convertir coordenadas del canvas a coordenadas de la imagen original
        h, w = self.original_image.shape[:2]
        canvas_w, canvas_h = int(w * self.scale), int(h * self.scale)
        offset_x = (1000 - canvas_w) // 2  # Ajustar por centrado en el canvas
        offset_y = 150 - canvas_h // 2

        x1 = int((self.roi_coords[0] - offset_x) / self.scale)
        y1 = int((self.roi_coords[1] - offset_y) / self.scale)
        x2 = int((self.roi_coords[2] - offset_x) / self.scale)
        y2 = int((self.roi_coords[3] - offset_y) / self.scale)

        # Asegurarse de que las coordenadas estén dentro de los límites
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

        if x1 >= x2 or y1 >= y2:
            messagebox.showerror("Error", "Invalid ROI dimensions")
            return

        # Recortar la imagen
        cropped_image = self.original_image[y1:y2, x1:x2]
        self.display_image(cropped_image, scale=1.0)  # Mostrar la imagen recortada
        self.processed_image = cropped_image  # Guardar como imagen procesada
        self.status.config(text=f"ROI cropped: ({x1}, {y1}, {x2}, {y2})")

    def process_image(self):
        """Procesa la imagen según el modo seleccionado:
            - 'procesar': Encuentra centroides de objetos circulares
            - 'recortar': Extrae círculos de la imagen
            """
    
        # Validación inicial: Verifica si hay una imagen cargada
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first")
            return

        try:
            # =============================================
            # 1. OBTENER PARÁMETROS DE LOS CONTROLES DE UI
            # =============================================
            area_min = self.area_min.get()          # Área mínima para detección
            area_max = self.area_max.get()          # Área máxima para detección
            threshold_mode = self.threshold_mode.get()  # Modo de threshold (otsu/manual)
            manual_thresh = self.manual_threshold.get() # Valor de threshold manual
            gaussian_k = self.gaussian_kernel.get()     # Tamaño del kernel gaussiano
            bilateral_d = self.bilateral_d.get()        # Parámetro d para filtro bilateral
            apply_erosion = self.apply_erosion.get()    # Bandera para aplicar erosión
            apply_dilation = self.apply_dilation.get()  # Bandera para aplicar dilatación
            morph_k = self.morph_kernel.get()           # Tamaño del kernel morfológico
            clahe_clip = self.clahe_clip.get()          # Límite de clip para CLAHE
            clahe_tile = self.clahe_tile.get()          # Tamaño de tile para CLAHE

            # Actualizar estado en la interfaz
            self.status.config(text="Processing image...")
            self.root.update()  # Forzar actualización de la UI
        
            # =============================================
            # 2. PROCESAMIENTO SEGÚN MODO SELECCIONADO
            # =============================================
            mode = self.mode.get()
        
            if mode == "procesar":
             # -----------------------------------------
             # MODO: PROCESAR (DETECCIÓN DE CENTROIDES)
             # -----------------------------------------
            
                # Determinar valor de threshold según modo seleccionado
                threshold_value = manual_thresh if threshold_mode == "manual" else 100
            
                # Paso 1: Detectar y centrar el círculo principal
                centered_image, radius, new_cx, new_cy = recortar_circulo(self.original_image,clahe_clip=2.0, clahe_tile=16)
            
                # Paso 2: Procesar contornos con K-Means
                (imagen_kmeans, mascara, centros_kmeans, tamano_imagen, imagen_contornos, imagen_filtrada) = procesar_contornos_kmeans(
                     self.original_image,
                     clahe_clip=clahe_clip,
                     clahe_tile=clahe_tile,
                     threshold_value=threshold_value,
                     area_min=area_min,
                     area_max=area_max,
                     kernel_size=morph_k,
                     radio=radius,
                     x=new_cx,
                     y=new_cy
                     )
            
                # Mostrar resultados
                self.display_image(imagen_kmeans, scale=self.scale_var.get())
            
                # Almacenar resultados para posible guardado
                self.processed_image = imagen_kmeans
                self.processed_results = {
                    'kmeans': imagen_kmeans,
                    'mascara': mascara,
                    'contornos': imagen_contornos,
                    'filtrada': imagen_filtrada,
                    'cx': new_cx,
                    'cy': new_cy,
                    'radio': radius,
                    'centros_kmeans': centros_kmeans,
                    'tamano_imagen': tamano_imagen
                    }
            
                # Actualizar estado
                status_text = f"Processing complete. {len(centros_kmeans)} centroides, tamaño: {tamano_imagen}"
                self.status.config(text=status_text)
            
                # Debug: Mostrar información en consola
                print("Centroides encontrados:", len(centros_kmeans))
                print("Centroides K-Means:", centros_kmeans)
                print("Tamaño de la imagen:", tamano_imagen)

            elif mode == "recortar":
            # -----------------------------------------
            # MODO: RECORTAR (EXTRACCIÓN DE CÍRCULOS)
            # -----------------------------------------
            
                # Paso 1: Preprocesamiento de la imagen
                imagen_gris = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(imagen_gris, (gaussian_k, gaussian_k), 0)
            
                # Paso 2: Mejora de contraste (CLAHE)
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
                img_clahe = clahe.apply(blurred)
            
                # Paso 3: Filtrado bilateral (reduce ruido preservando bordes)
                imagen_bilateral = cv2.bilateralFilter(img_clahe, bilateral_d, bilateral_d, 75)

                # Paso 4: Thresholding (binarización)
                if threshold_mode == "otsu":
                    _, th1 = cv2.threshold(imagen_bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    _, th1 = cv2.threshold(imagen_bilateral, manual_thresh, 255, cv2.THRESH_BINARY)

                # Paso 5: Operaciones morfológicas
                kernel = np.ones((morph_k, morph_k), np.uint8)
                if apply_erosion:
                    th1 = cv2.erode(th1, kernel, iterations=1)
                if apply_dilation:
                    th1 = cv2.dilate(th1, kernel, iterations=1)

                # Paso 6: Detección de contornos
                contours, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                # Paso 7: Análisis de contornos
                (imagen_areas, imagen_filtrada, centroides) = analizar_contornos(imagen_gris, contours, area_min, area_max)
            
                # Paso 8: Filtrado y recorte de círculos
                (imagep, circulos, cropped_images, mask) = filtrar_y_recortar_circulos(
                     self.original_image, 
                     contours, 
                     centroides, 
                     "output"
                     )

                # Mostrar y almacenar resultados
                self.display_image(imagep, scale=self.scale_var.get())
                self.processed_image = imagep
                self.processed_results = {
                    'areas': imagen_areas,
                    'filtered': imagen_filtrada,
                    'final': imagep,
                    'circles': circulos,
                    'mask': mask,
                    'cropped': cropped_images
                    }
                self.status.config(text="Processing complete")

        except Exception as e:
            # Manejo de errores
            error_msg = f"Processing failed: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status.config(text="Error during processing")
            
            
    def save_results(self):
        """Guarda los resultados del procesamiento de imágenes en un directorio seleccionado por el usuario.
    
    Los archivos guardados dependen del modo de operación:
    - Modo 'recortar': Guarda imágenes recortadas de círculos detectados
    - Modo 'procesar': Guarda imágenes intermedias y datos de centroides en JSON
    """
        # Verificar si hay resultados para guardar
        if not hasattr(self, 'processed_results'):
            messagebox.showerror("Error", "Please process an image first")
            return

        save_dir = filedialog.askdirectory()
        if save_dir:
            try:
                # Guardar resultados del procesamiento si existen
                if hasattr(self, 'processed_results'):
                    # MODO RECORTAR: Guardar imágenes de círculos recortados
                    if self.mode.get() == "recortar":
                        #cv2.imwrite(os.path.join(save_dir, "areas_image.jpg"), self.processed_results['areas'])
                        #cv2.imwrite(os.path.join(save_dir, "filtered_image.jpg"), self.processed_results['filtered'])
                        #cv2.imwrite(os.path.join(save_dir, "final_image.jpg"), self.processed_results['final'])
                        #cv2.imwrite(os.path.join(save_dir, "circles_image.jpg"), self.processed_results['circles'])
                        #cv2.imwrite(os.path.join(save_dir, "mask_image.jpg"), self.processed_results['mask'])
                                        
                        for i, cropped_img in enumerate(self.processed_results['cropped'], 1):
                            output_path = os.path.join(save_dir, f"cropped_{i}.jpg")
                            cv2.imwrite(output_path, cropped_img)
                
                    
                    else:
                        # Guardar imágenes de cada etapa del procesamiento
                        cv2.imwrite(os.path.join(save_dir, "kmeans.jpg"), self.processed_results['kmeans'])
                        cv2.imwrite(os.path.join(save_dir, "mascara.jpg"), self.processed_results['mascara'])
                        cv2.imwrite(os.path.join(save_dir, "imagen_contornos.jpg"), self.processed_results['contornos'])
                        cv2.imwrite(os.path.join(save_dir, "imagen_filtrada.jpg"), self.processed_results['filtrada'])
                        # Guardar coordenadas y tamaño en JSON
                        datos = {
                            "centros_kmeans": [{"x": cx, "y": cy} for cx, cy in self.processed_results['centros_kmeans']],
                            "tamano_imagen": {
                                "ancho": self.processed_results['tamano_imagen'][0],
                                "alto": self.processed_results['tamano_imagen'][1]
                                },
                            "centro_imagen": {
                                "cx": self.processed_results['cx'],
                                "cy": self.processed_results['cy']
                                },
                            "radio": self.processed_results['radio']
                            }
                        with open(os.path.join(save_dir, "resultados.json"), "w") as f:
                            json.dump(datos, f, indent=4)
                            
                # Guardar el ROI recortado si existe
                if self.processed_image is not None and self.roi_coords is not None:
                    roi_path = os.path.join(save_dir, "roi_cropped.jpg")
                    cv2.imwrite(roi_path, self.processed_image)
                    self.status.config(text=f"ROI saved to: {roi_path}")
                
                messagebox.showinfo("Success", "Results saved successfully")
                self.status.config(text=f"Results saved to: {save_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {str(e)}")
                self.status.config(text="Error during save")

def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()