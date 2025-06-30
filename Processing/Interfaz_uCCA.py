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
        self.roi_rect = None  # ID of the rectangle on the canvas
        self.roi_start = None  # Initial coordinates of the ROI
        self.roi_coords = None  # Current coordinates of the ROI (x1, y1, x2, y2)
        
        # Create GUI elements
        self.create_widgets()
        # Bind mode change to a function
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
        """Update area parameters (values and ranges) according to the selected mode"""
        current_mode = self.mode.get()
        
        if current_mode == "process":
            ## Lower ranges and values for "process" mode
            self.area_min.config(from_=0, to=500)    # Example: maximum limit of 5000
            self.area_min.set(50)                     # Lower initial value
            self.area_max.config(from_=0, to=5000)    
            self.area_max.set(2000)                  
        elif current_mode == "crop":
            # Original ranges and values for "crop" mode
            self.area_min.config(from_=0, to=150000)  # Original limit
            self.area_min.set(100)                    # Original initial value
            self.area_max.config(from_=150000, to=300000) 
            self.area_max.set(5000)                   

    def update_scale_with_keys(self, event):
        """Adjust the scale with the left/right keys."""
        step = 0.05  # Incremento o decremento
        current_value = self.scale_var.get()

        if event.keysym == "Left":
            new_value = max(self.scale_slider.cget("from"), current_value - step)
        elif event.keysym == "Right":
            new_value = min(self.scale_slider.cget("to"), current_value + step)

        self.scale_var.set(new_value)  #  Refresh slider 
        
    def display_image(self, img, scale=1.0):
        """Update the displayed image"""
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
        self.scale = scale  

    def update_image_scale(self, value):
        """Update the crop image."""
        if self.original_image is not None:
            scale = self.scale_var.get()
            self.display_image(self.original_image, scale)
            if self.roi_coords:  # Redraw ROI if it exist
                self.redraw_roi()

    def load_image(self):
        """Load image"""
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
            self.roi_coords = None  # Reset ROI when loading a new image
            
    def update_roi(self, event):
        """Update the size of the ROI while dragging the mouse."""
        if self.roi_start:
            self.canvas.coords(self.roi_rect, self.roi_start[0], self.roi_start[1], event.x, event.y)


    def start_roi(self, event):
        """Start ROI selection on click."""
        if self.original_image is None:
            return
        self.roi_start = (event.x, event.y)
        if self.roi_rect:
            self.canvas.delete(self.roi_rect)
        self.roi_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="red", width=2
        )
        

    def finalize_roi(self, event):
        """End ROI selection on mouse release."""
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
        """Redraw ROI to change image scale."""
        if self.roi_coords and self.roi_rect:
            self.canvas.delete(self.roi_rect)
            self.roi_rect = self.canvas.create_rectangle(
                self.roi_coords[0], self.roi_coords[1], self.roi_coords[2], self.roi_coords[3],
                outline="red", width=2
            )

    def crop_roi(self):
        """Crop the region selected by the ROI and display it."""
        if self.original_image is None or self.roi_coords is None:
            messagebox.showerror("Error", "Please load an image and select a ROI first")
            return

        # Convert canvas coordinates to original image coordinates

        h, w = self.original_image.shape[:2]
        canvas_w, canvas_h = int(w * self.scale), int(h * self.scale)
        offset_x = (1000 - canvas_w) // 2  # Fit by centering on the canvas
        offset_y = 150 - canvas_h // 2

        x1 = int((self.roi_coords[0] - offset_x) / self.scale)
        y1 = int((self.roi_coords[1] - offset_y) / self.scale)
        x2 = int((self.roi_coords[2] - offset_x) / self.scale)
        y2 = int((self.roi_coords[3] - offset_y) / self.scale)

        # Make sure that the coordinates are within the boundaries
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

        if x1 >= x2 or y1 >= y2:
            messagebox.showerror("Error", "Invalid ROI dimensions")
            return

        # Crop image
        cropped_image = self.original_image[y1:y2, x1:x2]
        self.display_image(cropped_image, scale=1.0)  # Show crop image
        self.processed_image = cropped_image  # Save as image process
        self.status.config(text=f"ROI cropped: ({x1}, {y1}, {x2}, {y2})")

    def process_image(self):
        """
        Process the image according to the selected mode:
            - 'process': Find centroids of circular objects
            - 'crop': Extract circles from the image
        """
    
        # Initial validation: Check if an image is loaded
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first")
            return

        try:
            # =============================================
            # 1. GET PARAMETERS FROM UI CONTROLS
            # =============================================
            
            area_min = self.area_min.get()          # Minimum area for detection
            area_max = self.area_max.get()          # Maximum area for detection
            threshold_mode = self.threshold_mode.get()  # Threshold mode (otsu/manual)
            manual_thresh = self.manual_threshold.get() # Manual threshold value
            gaussian_k = self.gaussian_kernel.get()     # Gaussian kernel size
            bilateral_d = self.bilateral_d.get()        # Parameter d for bilateral filter
            apply_erosion = self.apply_erosion.get()    # Flag to apply erosion
            apply_dilation = self.apply_dilation.get()  # Flag to apply dilation
            morph_k = self.morph_kernel.get()           # Morphological kernel size
            clahe_clip = self.clahe_clip.get()          # Clip limit for CLAHE
            clahe_tile = self.clahe_tile.get()          # Tile size for CLAHE


            # Update status in the interface
            self.status.config(text="Processing image...")
            self.root.update()  # Force UI update
        
            # =============================================
            # 2. PROCESSING ACCORDING TO SELECTED MODE
            # =============================================
            mode = self.mode.get()
        
            if mode == "process":
             # -----------------------------------------
             # MODO: PROCESS (CENTROID DETECTION)
             # -----------------------------------------
            
                # Determine threshold value according to selected mode
                threshold_value = manual_thresh if threshold_mode == "manual" else 100
            
                # Step 1: Detect and center the main circle
                centered_image, radius, new_cx, new_cy = recortar_circulo(self.original_image,clahe_clip=2.0, clahe_tile=16)
            
                # Step 2: Process contours with K-Means
                (kmeans_image, mask, kmeans_centers, image_size, contour_image, filtered_image) = procesar_contornos_kmeans(
                    self.original_image,
                    clahe_clip=clahe_clip,
                    clahe_tile=clahe_tile,
                    threshold_value=threshold_value,
                    area_min=area_min,
                    area_max=area_max,
                    kernel_size=morph_k,
                    radius=radius,
                    x=new_cx,
                    y=new_cy
                )

            
                # Display results
                self.display_image(kmeans_image, scale=self.scale_var.get())
            
                # Store results for possible saving

                self.processed_image = kmeans_image
                self.processed_results = {             
                    'kmeans': kmeans_image,
                    'mask': mask,
                    'contours': contour_image,
                    'filtered': filtered_image,
                    'cx': new_cx,
                    'cy': new_cy,
                    'radius': radius,
                    'kmeans_centers': kmeans_centers,
                    'image_size': image_size
                    }
            
                
                # Update status
                status_text = f"Processing complete. {len(kmeans_centers)} centroids, size: {image_size}"
                self.status.config(text=status_text)
    
                # Debug: Display information in console
                print("Centroids found:", len(kmeans_centers))
                print("K-Means Centroids:", kmeans_centers)
                print("Image size:", image_size


            elif mode == "crop":
            # -----------------------------------------
            # MODO: CROP (CIRCLE EXTRACTION)
            # -----------------------------------------
            
                # Step 1: Image preprocessing
                gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray_image, (gaussian_k, gaussian_k), 0)
            
                #Step 2: Contrast enhancement (CLAHE)
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
                img_clahe = clahe.apply(blurred)
            
                # Step 3: Bilateral filtering (reduces noise while preserving edges
                bilateral_image = cv2.bilateralFilter(img_clahe, bilateral_d, bilateral_d, 75)

                # Step 4: Thresholding (binarization)
                if threshold_mode == "otsu":
                    _, th1 = cv2.threshold(bilateral_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    _, th1 = cv2.threshold(bilateral_image, manual_thresh, 255, cv2.THRESH_BINARY)

                # Step 5: Morphological operations

                kernel = np.ones((morph_k, morph_k), np.uint8)
                if apply_erosion:
                    th1 = cv2.erode(th1, kernel, iterations=1)
                if apply_dilation:
                    th1 = cv2.dilate(th1, kernel, iterations=1)

                # Step 6: Contour detection

                contours, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                # Step 7: Contour analysis
                (image_areas, filtered_image, centroids) = analizar_contornos(gray_image, contours, area_min, area_max)
            
                # Step 8: Circle filtering and cropping
                (imagep, circles, cropped_images, mask) = filtrar_y_recortar_circulos(
                     self.original_image, 
                     contours, 
                     centroids, 
                     "output"
                     )

                # Display and store results
                self.display_image(imagep, scale=self.scale_var.get())
                self.processed_image = imagep
                self.processed_results = {                
                    'areas': image_areas,
                    'filtered': filtered_image,
                    'final': imagep,
                    'circles': circles,
                    'mask': mask,
                    'cropped': cropped_images
                    }
                self.status.config(text="Processing complete")

        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status.config(text="Error during processing")
            
            
    def save_results(self):
        """
        Saves the image processing results in a directory selected by the user.

        The saved files depend on the operation mode:
        - 'crop' mode: Saves cropped images of detected circles
        - 'process' mode: Saves intermediate images and centroid data in JSON
    """
        # Check if there are results to save

        if not hasattr(self, 'processed_results'):
            messagebox.showerror("Error", "Please process an image first")
            return

        save_dir = filedialog.askdirectory()
        if save_dir:
            try:
                # Save processing results if they exist
                if hasattr(self, 'processed_results'):
                    #CROP MODE: Save cropped circle images
                    if self.mode.get() == "recortar":
     
                        for i, cropped_img in enumerate(self.processed_results['cropped'], 1):
                            output_path = os.path.join(save_dir, f"cropped_{i}.jpg")
                            cv2.imwrite(output_path, cropped_img)
                
                    
                    else:
                        # Save images from each processing stage
                        cv2.imwrite(os.path.join(save_dir, "kmeans.jpg"), self.processed_results['kmeans'])
                        cv2.imwrite(os.path.join(save_dir, "mask.jpg"), self.processed_results['mask'])
                        cv2.imwrite(os.path.join(save_dir, "contours_image.jpg"), self.processed_results['contours'])
                        cv2.imwrite(os.path.join(save_dir, "filtered_image.jpg"), self.processed_results['filtered'])
                        # Save coordinates and size in JSON
                        datos = {
                            "kmeans_centers": [{"x": cx, "y": cy} for cx, cy in self.processed_results['centros_kmeans']],
                            "image_size": {
                                "width": self.processed_results['image_size'][0],
                                "height": self.processed_results['image_size'][1]
                                },
                            "image_center": {
                                "cx": self.processed_results['cx'],
                                "cy": self.processed_results['cy']
                                },
                            "radius": self.processed_results['radius']
                            }
                        with open(os.path.join(save_dir, "results.json"), "w") as f:
                            json.dump(data, f, indent=4)
                            
                # the cropped ROI if it exists
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
