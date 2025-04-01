# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 23:03:38 2025

@author: Abraham
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import csv

def classify_centroids(centers, x_center, y_center, r):
    """
    Clasifica los centroides en dos listas: dentro o fuera de un círculo de radio r.

    Parámetros:
    - centers: Lista de diccionarios con coordenadas {'x': val, 'y': val}
    - x_center, y_center: Coordenadas del centro del círculo
    - r: Radio del círculo

    Retorna:
    - inside_centroids: Lista de centroides dentro del círculo
    - outside_centroids: Lista de centroides fuera del círculo
    """
    inside_centroids = []
    outside_centroids = []

    for center in centers:
        x, y = center['x'], center['y']
        distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

        centroid_data = {'x': x, 'y': y, 'distance': distance}

        if distance <= r:
            inside_centroids.append(centroid_data)
        else:
            outside_centroids.append(centroid_data)

    return inside_centroids, outside_centroids

def match_points(experimental_points, theoretical_points):
    """
    Empareja cada punto experimental con el punto teórico más cercano.

    Parámetros:
    - experimental_points: Lista de puntos experimentales [{'x': x, 'y': y, ...}]
    - theoretical_points: Lista de puntos teóricos [{'x': x, 'y': y, ...}]

    Retorna:
    - matched_pairs: Lista de tuplas [(exp_point, theo_point), ...]
    """
    matched_pairs = []

    for exp_point in experimental_points:
        min_distance = float('inf')
        closest_theo_point = None

        # Encontrar el punto teórico más cercano
        for theo_point in theoretical_points:
            distance = np.sqrt((theo_point['x'] - exp_point['x']) ** 2 + (theo_point['y'] - exp_point['y']) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_theo_point = theo_point

        matched_pairs.append((exp_point, closest_theo_point))

    return matched_pairs


def compute_errors(matched_pairs):
    """
    Calcula las diferencias entre puntos experimentales y teóricos.

    Parámetros:
    - matched_pairs: Lista de tuplas [(exp_point, theo_point), ...]

    Retorna:
    - errors: Lista de diccionarios con errores [{'error_x': ..., 'error_y': ..., 'error_dist': ...}]
    """
    errors = []

    for exp_point, theo_point in matched_pairs:
        error_x = theo_point['x'] - exp_point['x']
        error_y = theo_point['y'] - exp_point['y']
        error_dist = np.sqrt(error_x**2 + error_y**2)

        errors.append({
            'error_x': error_x,
            'error_y': error_y,
            'error_dist': error_dist
        })

    return errors



def save_to_csv(matched_pairs, errors, filename="resultados.csv"):
    """
    Guarda los puntos emparejados y los errores en un archivo CSV.

    Parámetros:
    - matched_pairs: Lista de tuplas [(exp_point, theo_point), ...]
    - errors: Lista de diccionarios con errores [{'error_x': ..., 'error_y': ..., 'error_dist': ...}]
    - filename: Nombre del archivo CSV (por defecto: "resultados.csv")
    """
    # Abrir el archivo CSV en modo escritura
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Escribir el encabezado del CSV
        writer.writerow([
            "Experimental X", "Experimental Y",
            "Teórico X", "Teórico Y",
            "Error X", "Error Y", "Error Distancia"
        ])
        
        # Escribir los datos
        for (exp_point, theo_point), error in zip(matched_pairs, errors):
            writer.writerow([
                exp_point['x'], exp_point['y'],
                theo_point['x'], theo_point['y'],
                error['error_x'], error['error_y'],f"{ error['error_dist']:.2f}"
            ])
    print(f"Los resultados se han guardado en '{filename}'.")

# === CARGA DE DATOS JSON ===
with open('resultados7.json', 'r') as f:
    data = json.load(f)

centers = data['centros_kmeans']
tamano_imagen = data['tamano_imagen']
centro_imagen = data['centro_imagen']

size = tamano_imagen['alto']  
x_center = centro_imagen['cx']  
y_center = centro_imagen['cy']  
r = 206 -10

# Clasificar puntos experimentales dentro y fuera del círculo
inside_experiment, outside_experiment = classify_centroids(centers, x_center, y_center, r)    

        
# Create blank image
imagen = np.zeros((size, size), dtype=np.uint8)
circle_mask = np.zeros((size, size), dtype=np.uint8)
cv2.circle(circle_mask, (x_center, y_center), r, 255, -1)  # Main circle mask
# Set radius for smaller circles (adjusted for new image size)
radius = 8  # Adjust this value as needed

# Draw circles from JSON centers
for center in centers:
    x, y = center['x'], center['y']
    # Dibujar un pequeño circuloen cada punto
    cv2.circle(imagen, (x, y), radius, 255, -1)


inside = cv2.bitwise_and(imagen, circle_mask)  # Intersección

cv2.circle(imagen, (x_center, y_center), r, 255, 2)
cv2.circle(inside, (x_center, y_center), r, 255, 2)  # contornos circulo en inside 


# === GENERACIÓN DE CÍRCULOS TEÓRICOS ===
#N = 7
#N = 5 
#N = 14
N = 12
grid_size = size // N
radius = 8  
centros_teoricos = []
# Crear una imagen en blanco
image = np.zeros((size, size), dtype=np.uint8)

for i in range(N):
    for j in range(N):
        x = int((i + 0.5) * grid_size)
        y = int((j + 0.5) * grid_size)
        centros_teoricos.append({'x': x, 'y': y})
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if dx**2 + dy**2 <= radius**2:
                    xi, yi = int(x + dx), int(y + dy)
                    if 0 <= xi < size and 0 <= yi < size:
                        image[yi, xi] = 255

inside_theoretical, outside_theoretical = classify_centroids(centros_teoricos, x_center, y_center, r)

######----------------------------------------------
circulo = image.copy()
cv2.circle(circulo, (x_center, y_center), r, 255, 2)
#####-------------------------------------------------
inside_circles = cv2.bitwise_and(image, circle_mask)  # Intersección
cv2.circle(inside_circles, (x_center, y_center), r, 255, 2)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(inside, cmap="gray")
plt.title("Centroides experimentales")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(inside_circles, cmap = "gray")
plt.title("Centroides teóricos")
plt.axis('off')

plt.show()



# === CÁLCULO DE ERRORES ===
matched_pairs = match_points(inside_experiment, inside_theoretical)
errors = compute_errors(matched_pairs)
# === Guardar resultados en CSV ===
save_to_csv(matched_pairs, errors, filename="resultados.csv") 
# Calcular el promedio de las distancias
total_distance = sum(error['error_dist'] for error in errors)  # Suma de todas las distancias
avg = total_distance / len(errors)  # Promedio

# === IMPRIMIR RESULTADOS ===
print("\n###########------ Comparación Teórico vs. Experimental ------##############")
for i, ((exp, theo), err) in enumerate(zip(matched_pairs, errors)):
    print(f"Punto {i+1}:")
    print(f"Teórico:       ({theo['x']}, {theo['y']})")
    print(f"Experimental:  ({exp['x']}, {exp['y']})")
    print(f"Error: Δx={err['error_x']:.2f}, Δy={err['error_y']:.2f}, Dist={err['error_dist']:.2f}")
    print("-" * 60)

# === VISUALIZACIÓN ===
imagen = np.zeros((size, size), dtype=np.uint8)
for center in inside_theoretical:
    cv2.circle(imagen, (center['x'], center['y']), radius, 255, -1)

cv2.circle(imagen, (x_center, y_center), r, 255, 2)  

# Crear una figura
plt.figure(figsize=(8, 8))
plt.imshow(imagen, cmap="gray")
# Graficar puntos experimentales
plt.scatter(
    [p['x'] for p in inside_experiment],
    [p['y'] for p in inside_experiment],
    color='red', label='Experimental', marker='o'
)

# Graficar puntos teóricos
plt.scatter(
    [p['x'] for p in inside_theoretical],
    [p['y'] for p in inside_theoretical],
    color='blue', label='Teórico', marker='x'
)

# Dibujar líneas que conecten los puntos emparejados (para mostrar errores)
for exp_point, theo_point in matched_pairs:
    plt.plot(
        [exp_point['x'], theo_point['x']],
        [exp_point['y'], theo_point['y']],
        color='yellow', linestyle='--', linewidth=0.8
    )

# Agregar etiquetas a los puntos
for i, (exp_point, theo_point) in enumerate(matched_pairs):
    plt.text(exp_point['x'], exp_point['y'], f"{i+1}", color='white', fontsize=9, ha='right', va='bottom')
    plt.text(theo_point['x'], theo_point['y'], f"{i+1}", color='white', fontsize=9, ha='left', va='top')

# Configuración del gráfico
plt.title("Comparación de Centroides Experimentales vs. Teóricos")

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('off')
#plt.gca().set_aspect('equal', adjustable='box')  # Escala igual en X e Y

# Mostrar el gráfico
plt.show()

