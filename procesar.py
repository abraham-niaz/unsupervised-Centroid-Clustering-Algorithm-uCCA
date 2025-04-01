# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 11:26:38 2025

@author: Abraham
"""

import cv2
import numpy as np
import os


def analizar_contornos(image, contours, area_min=70000, area_max=400000, debug=False):
    """
    Analiza y filtra contornos según su área, dibujándolos con información adicional.
    
    Parametros:
    - image: Imagen de entrada en escala de grises
    - contours: Lista de contornos obtenidos de cv2.findContours
    - area_min: Área mínima para filtrar contornos (default: 70000)
    - area_max: Área máxima para filtrar contornos (default: 400000)
    - debug: Si es True, retorna información adicional para depuración
    
    Returns:
    - imagen_areas: Imagen con contornos numerados y centroides
    - imagen_filtrada: Imagen con contornos filtrados coloreados
    - dict: (opcional) Información adicional si debug=True
    """
    # Calcular y ordenar áreas de todos los contornos
    areas_contornos = [cv2.contourArea(contour) for contour in contours]
    areas_contornos.sort(reverse=True)
    
    # Crear copias de la imagen para visualización
    imagen_areas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    imagen_filtrada = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Listas para almacenar resultados
    areas_filtradas = []
    centroides = []
    
    # Procesar cada contorno
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filtrar por área mínima y máxima
        if area_min <= area <= area_max:
            areas_filtradas.append((i + 1, area))
            
            # Generar color aleatorio
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Dibujar contorno filtrado
            cv2.drawContours(imagen_filtrada, [contour], -1, color, thickness=5)
            
            # Calcular centroide
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroides.append([cx, cy])
                
                # Dibujar número y centroide en imagen_areas
                cv2.putText(imagen_areas, f"{i+1}", (cx, cy), 
                          cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 3)
                cv2.circle(imagen_areas, (cx, cy), 15, (255, 255, 255), -1)
    
    # Imprimir información
    print("Áreas ordenadas de mayor a menor:", areas_contornos)
    print(f"Área mínima: {area_min}")
    print(f"Área máxima: {area_max}")
    print("Contornos Filtrados por Área")
    for num, area in areas_filtradas:
        print(f"Contorno {num}, Área = {area:.2f}")
    
    if debug:
        debug_info = {
            'areas_totales': areas_contornos,
            'areas_filtradas': areas_filtradas,
            'centroides': centroides
        }
        return imagen_areas, imagen_filtrada, debug_info
    
    return imagen_areas, imagen_filtrada, centroides

def filtrar_y_recortar_circulos(image, contours, centroides, carpeta_salida, tolerance=0.20, circularity_min=0.75, 
                               circularity_max=1.1, expansion_factor=0.15):
    """
    Filtra contornos aproximadamente circulares, calcula círculos promedio y recorta regiones de interés.

    Parameters:
    - image: Imagen de entrada en formato BGR.
    - contours: Lista de contornos obtenidos con cv2.findContours.
    - centroides: Lista de coordenadas [cx, cy] de los centroides de los contornos.
    - carpeta_salida: Directorio donde se guardarán las imágenes recortadas.
    - tolerance: Porcentaje de tolerancia para el rango de área (default: 0.20).
    - circularity_min: Umbral mínimo de circularidad (default: 0.75).
    - circularity_max: Umbral máximo de circularidad (default: 1.1).
    - expansion_factor: Factor para expandir el recorte más allá del radio (default: 0.15).

    Returns:
    - imagep: Imagen original con rectángulos dibujados alrededor de las regiones recortadas.
    - circulos: Imagen con los círculos promedio dibujados.
    - cropped_images: Lista de imágenes recortadas.
    - mask: Máscara con los círculos mínimos que encierran los contornos filtrados.
    """
    
    # Convertir a escala de grises si no lo está
    if len(image.shape) == 3:
        imagen_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = image

    # Calcular el área máxima de los contornos
    if not contours:
        print("No se encontraron contornos.")
        return image, np.zeros_like(imagen_gris), [], np.zeros_like(imagen_gris)
    max_area = max(cv2.contourArea(contour) for contour in contours)
    print("Máxima área:", max_area)

    # Definir el rango de tolerancia (20% del área máxima por defecto)
    min_area = max_area * (1 - tolerance)
    max_area_tolerance = max_area * (1 + tolerance)
    print("Área mínima:", min_area)
    print("Área máxima con tolerancia:", max_area_tolerance)

    # Crear una máscara en negro
    mask = np.zeros_like(imagen_gris)

    # Filtrar contornos y calcular radios
    filtered_contours = []
    radii = []  # Lista para almacenar los radios de los círculos

    for contour in contours:
        area_contour = cv2.contourArea(contour)
        
        # Filtrar por área dentro del rango de tolerancia
        if min_area <= area_contour <= max_area_tolerance:
            # Obtener el círculo mínimo que encierra el contorno
            (x, y), radius = cv2.minEnclosingCircle(contour)
            area_circle = np.pi * (radius ** 2)
            
            # Verificar si la forma es aproximadamente circular
            circularity = area_contour / area_circle  # Debe estar cerca de 1 para un círculo
            
            if circularity_min <= circularity <= circularity_max:
                filtered_contours.append(contour)
                radii.append(radius)
                cv2.circle(mask, (int(x), int(y)), int(radius), (255), thickness=3)

    # Calcular el radio promedio
    radio_promedio = int(np.mean(radii)) if radii else 0
    print("Radio promedio:", radio_promedio)

    # Dibujar círculos promedio en los centroides
    circulos = np.zeros_like(imagen_gris)
    for cx, cy in centroides:
        cv2.circle(circulos, (cx, cy), radio_promedio, (255, 0, 0), thickness=3)  # Azul para los círculos promedio

    # Lista para almacenar las imágenes recortadas
    cropped_images = []
    imagep = image.copy()  # Copia de la imagen original para dibujar rectángulos
    expansion = int(radio_promedio * expansion_factor)  # Margen adicional para el recorte

    # Recortar cada círculo promedio
    for i, (cx, cy) in enumerate(centroides):
        # Definir los límites del recorte
        x_min = max(0, cx - radio_promedio - expansion)
        x_max = min(imagep.shape[1], cx + radio_promedio + expansion)
        y_min = max(0, cy - radio_promedio - expansion)
        y_max = min(imagep.shape[0], cy + radio_promedio + expansion)

        # Recortar la región de la imagen
        cropped = image[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped)

        # Guardar la imagen recortada
        output_path = os.path.join(carpeta_salida, f"figura_filtrada_{i+1}.jpg")
        os.makedirs(carpeta_salida, exist_ok=True)  # Crear directorio si no existe
        cv2.imwrite(output_path, cropped)

        # Dibujar un rectángulo alrededor del área recortada en la imagen original
        cv2.rectangle(imagep, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=4)

    return imagep, circulos, cropped_images, mask

################################---------------##############################
def recortar_circulo(image, clahe_clip=2.0, clahe_tile=16):
    """
    Detecta y recorta el círculo de mayor radio en la imagen usando HoughCircles.
    
    Parameters:
    - image: Imagen de entrada en formato BGR.
    - clahe_clip: Límite de contraste para CLAHE (default: 1.0).
    - clahe_tile: Tamaño de la cuadrícula para CLAHE (default: 4).
    
    Returns:
    - output_image: Imagen original con el círculo detectado dibujado.
    - centered_image: Imagen con el círculo recortado centrado en un fondo negro.
    - radius: Radio del círculo detectado.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    img_clahe = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(img_clahe, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=20, maxRadius=300
    )
    
    output_image = image.copy()
    if circles is not None:
        max_circle = max(circles[0, :], key=lambda c: c[2])
        center = (int(max_circle[0]), int(max_circle[1]))
        radius = int(max_circle[2])
        cv2.circle(output_image, center, radius, (0, 255, 0), 3)
        
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        masked_image = cv2.bitwise_and(image, mask)
        
        x1, y1 = max(center[0] - radius, 0), max(center[1] - radius, 0)
        x2, y2 = min(center[0] + radius, image.shape[1]), min(center[1] + radius, image.shape[0])
        cropped_circle = masked_image[y1:y2, x1:x2]
        
        centered_image = np.zeros_like(image)
        ch, cw = centered_image.shape[:2]
        new_cx, new_cy = cw // 2, ch // 2
        h, w = cropped_circle.shape[:2]
        x_offset, y_offset = new_cx - w // 2, new_cy - h // 2
        centered_image[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_circle
        
        return centered_image, radius, new_cx, new_cy
    # Si no se detectan círculos, devolver valores por defecto o None
    return None, 0, 0, 0

def procesar_contornos_kmeans(image, threshold_value=100, area_min=100, area_max=5000, kernel_size=3, clahe_clip=1.0, clahe_tile=4, radio = 211, x =255, y =255):
    """
    Procesa la imagen para encontrar contornos, filtrarlos por área y aplicar K-Means a los centroides.
    
    Parameters:
    - image: Imagen de entrada en formato BGR.
    - threshold_value: Valor de umbral para binarización (default: 100).
    - area_min: Área mínima para filtrar contornos (default: 100).
    - area_max: Área máxima para filtrar contornos (default: 5000).
    - kernel_size: Tamaño del kernel para operaciones morfológicas (default: 3).
    - clahe_clip: Límite de contraste para CLAHE (default: 1.0).
    - clahe_tile: Tamaño de la cuadrícula para CLAHE (default: 4).
    
    Returns:
    - imagen_contornos: Imagen con todos los contornos dibujados.
    - imagen_filtrada: Imagen con contornos filtrados por área.
    - imagen_kmeans: Imagen con centroides agrupados por K-Means.
    """
    # Obtener el tamaño de la imagen
    alto, ancho = image.shape[:2]  # Extraer alto y ancho (ignora canales si los hay)
    tamano_imagen = (ancho, alto)  # Devolver como (width, height)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    img_clahe = clahe.apply(gray)
    
    imagen_suavizada = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    imagen_bilateral = cv2.bilateralFilter(imagen_suavizada, 7, 7, 75)
    _, binary = cv2.threshold(imagen_bilateral, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=1)
    bordes_cerrados = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    bordes_limpios = cv2.morphologyEx(bordes_cerrados, cv2.MORPH_OPEN, kernel)
    
    contornos, _ = cv2.findContours(bordes_limpios, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imagen_contornos = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), thickness=1)
    
    imagen_filtrada = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    centroides = []
    for i, contorno in enumerate(contornos):
        area = cv2.contourArea(contorno)
        if area_min <= area <= area_max:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.drawContours(imagen_filtrada, [contorno], -1, color, thickness=1)
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroides.append([cx, cy])
                cv2.putText(imagen_filtrada, f"{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    imagen_kmeans = image.copy()
    centros_kmeans = []
    mascara = np.zeros_like(image, dtype=np.uint8)
    #cv2.circle(mascara, (x, y), radio, (255, 255, 255), 2)  # Dibujar el círculo en la máscara
    if len(centroides) > 1:
        k = min(500, len(centroides))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(np.array(centroides, dtype=np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Dibujar los centros detectados en la imagen
        for center in centers:
            cx, cy = int(center[0]), int(center[1])
            centros_kmeans.append([cx, cy])  # Guardar las coordenadas
            cv2.circle(imagen_kmeans, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(mascara, (cx, cy), 5, (255, 255, 255), -1)
        
    
    return imagen_kmeans, mascara, centros_kmeans, tamano_imagen, imagen_contornos, imagen_filtrada