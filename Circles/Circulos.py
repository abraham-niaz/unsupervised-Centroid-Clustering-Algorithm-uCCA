#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    This code constitutes an important tool for obtaining 
#    the results in the manuscript entitled  "Machine Learning K−means 
#    algorithm applied to wavefront sensing in Bi-Ronchi/Hartmann tests 
#    with SLM", submitted to Optik de Elsevier.
#
#    Correspondings Authors:
#    Dr. Jesus Alonso Arriaga Hernandez
#    jesus.arriagahdz@correo.buap.mx;    dr.j.a.arriaga.hernandez@gmail.com
#    
#    Dra. Bolivia Teresa Cuevas Otahola
#    b.cuevas@irya.unam.mx;                      b.cuevas.otahola@gmail.com
#
#    Lic. Abraham Gilberto Díaz Nayotl
#    dn223470444@alm.buap.mx;                        gil.diaz1205@gmail.com
#
#
#    This code performs a comparative analysis between experimental and 
#    theoretical centroids in an image. It first loads the experimental 
#    data from a JSON file, extracts the centroids, and classifies which
#    ones are inside or outside a circle of a defined radius. Then, it 
#    generates a grid of theoretical points distributed uniformly, classifying 
#    them in the same way. Next, each experimental point is paired with its 
#    closest theoretical point by calculating Euclidean distances, computing 
#    coordinate and distance errors, and storing the results in a CSV file. 
#    Finally, the program visualizes and compares both sets of points (experimental 
#    and theoretical), drawing lines that represent the errors between them 
#    and displaying the average difference.

import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import csv

def classify_centroids(centers, x_center, y_center, r):
    
    """
    Classify centroids into two lists: inside or outside a circle with radius r.

    Parameters:
    - centers: List of dictionaries with coordinates {'x': val, 'y': val}
    - x_center, y_center: Coordinates of the circle's center
    - r: Radius of the circle

    Returns:
    - inside_centroids: List of centroids inside the circle
    - outside_centroids: List of centroids outside the circle
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
    Match each experimental point with the nearest theoretical point.

    Parameters:
    - experimental_points: List of experimental points [{'x': x, 'y': y, ...}]
    - theoretical_points: List of theoretical points [{'x': x, 'y': y, ...}]

    Returns:
    - matched_pairs: List of tuples [(exp_point, theo_point), ...]
    """

    matched_pairs = []

    for exp_point in experimental_points:
        min_distance = float('inf')
        closest_theo_point = None

        # Find the nearest theoretical point
        
        for theo_point in theoretical_points:
            distance = np.sqrt((theo_point['x'] - exp_point['x']) ** 2 + (theo_point['y'] - exp_point['y']) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_theo_point = theo_point

        matched_pairs.append((exp_point, closest_theo_point))

    return matched_pairs


def compute_errors(matched_pairs):
    
    """
    Calculate the differences between experimental and theoretical points.

    Parameters:
    - matched_pairs: List of tuples [(exp_point, theo_point), ...]

    Returns:
    - errors: List of dictionaries with errors [{'error_x': ..., 'error_y': ..., 'error_dist': ...}]
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



def save_to_csv(matched_pairs, errors, filename="results.csv"):
    
    """
    Save the matched points and errors to a CSV file.

    Parameters:
    - matched_pairs: List of tuples [(exp_point, theo_point), ...]
    - errors: List of dictionaries with errors [{'error_x': ..., 'error_y': ..., 'error_dist': ...}]
    - filename: Name of the CSV file (default: "results.csv")
    """

    # Open file CSV 
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the CSV header
        writer.writerow([
            "Experimental X", "Experimental Y",
            "Theoretical X", "Theoretical Y",
            "Error X", "Error Y", "Error Distance"
        ])
        
        # Write the data
        for (exp_point, theo_point), error in zip(matched_pairs, errors):
            writer.writerow([
                exp_point['x'], exp_point['y'],
                theo_point['x'], theo_point['y'],
                error['error_x'], error['error_y'],f"{ error['error_dist']:.2f}"
            ])
    print(f"The results have been saved in '{filename}'.")

# === Load data JSON ===
with open('resultados7.json', 'r') as f:
    data = json.load(f)

centers = data['centros_kmeans']
image_size = data['tamano_imagen']
image_center = data['centro_imagen']

size = image_size['alto']  
x_center = image_center['cx']  
y_center = image_center['cy']  
r = 206 - 10

# Classify experimental points inside and outside the circle
inside_experiment, outside_experiment = classify_centroids(centers, x_center, y_center, r)    

        
# Create blank image
image = np.zeros((size, size), dtype=np.uint8)
circle_mask = np.zeros((size, size), dtype=np.uint8)
cv2.circle(circle_mask, (x_center, y_center), r, 255, -1)  # Main circle mask
# Set radius for smaller circles (adjusted for new image size)
radius = 8  # Adjust this value as needed

# Draw circles from JSON centers
for center in centers:
    x, y = center['x'], center['y']
    # Draw a small circle at each point
    cv2.circle(image, (x, y), radius, 255, -1)


inside = cv2.bitwise_and(image, circle_mask)  # Intersection

cv2.circle(image, (x_center, y_center), r, 255, 2)
cv2.circle(inside, (x_center, y_center), r, 255, 2)  # Circle contours in inside 


# === GENERATION OF THEORETICAL CIRCLES ===
N = 12
grid_size = size // N
radius = 8  
theoretical_centers = []
# Create a blank image
image = np.zeros((size, size), dtype=np.uint8)


for i in range(N):
    for j in range(N):
        x = int((i + 0.5) * grid_size)
        y = int((j + 0.5) * grid_size)
        theoretical_centers.append({'x': x, 'y': y})
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if dx**2 + dy**2 <= radius**2:
                    xi, yi = int(x + dx), int(y + dy)
                    if 0 <= xi < size and 0 <= yi < size:
                        image[yi, xi] = 255

inside_theoretical, outside_theoretical = classify_centroids(theoretical_centers, x_center, y_center, r)


circles = cv2.bitwise_and(image, circle_mask)  # Intersection
cv2.circle(inside_circles, (x_center, y_center), r, 255, 2)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(inside, cmap="gray")
plt.title("Experimental Centroids")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(inside_circles, cmap="gray")
plt.title("Theoretical Centroids")
plt.axis('off')

plt.show()



# === ERROR CALCULATION ===
matched_pairs = match_points(inside_experiment, inside_theoretical)
errors = compute_errors(matched_pairs)
# === Save results to CSV ===
save_to_csv(matched_pairs, errors, filename="results.csv") 
# Calculate the average distance
total_distance = sum(error['error_dist'] for error in errors)  # Sum of all distances
avg = total_distance / len(errors)  # Average

# === PRINT RESULTS ===
print("\n###########------ Theoretical vs. Experimental Comparison ------##############")
for i, ((exp, theo), err) in enumerate(zip(matched_pairs, errors)):
    print(f"Point {i+1}:")
    print(f"Theoretical:       ({theo['x']}, {theo['y']})")
    print(f"Experimental:  ({exp['x']}, {exp['y']})")
    print(f"Error: Δx={err['error_x']:.2f}, Δy={err['error_y']:.2f}, Dist={err['error_dist']:.2f}")
    print("-" * 60)

# === VISUALIZATION ===
image = np.zeros((size, size), dtype=np.uint8)
for center in inside_theoretical:
    cv2.circle(image, (center['x'], center['y']), radius, 255, -1)

cv2.circle(image, (x_center, y_center), r, 255, 2)  

# Create a figure
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap="gray")
# Plot experimental points
plt.scatter(
    [p['x'] for p in inside_experiment],
    [p['y'] for p in inside_experiment],
    color='red', label='Experimental', marker='o'
)

# Plot theoretical points
plt.scatter(
    [p['x'] for p in inside_theoretical],
    [p['y'] for p in inside_theoretical],
    color='blue', label='Theoretical', marker='x'
)

# Draw lines connecting matched points (to show errors)
for exp_point, theo_point in matched_pairs:
    plt.plot(
        [exp_point['x'], theo_point['x']],
        [exp_point['y'], theo_point['y']],
        color='yellow', linestyle='--', linewidth=0.8
    )

# Add labels to points
for i, (exp_point, theo_point) in enumerate(matched_pairs):
    plt.text(exp_point['x'], exp_point['y'], f"{i+1}", color='white', fontsize=9, ha='right', va='bottom')
    plt.text(theo_point['x'], theo_point['y'], f"{i+1}", color='white', fontsize=9, ha='left', va='top')

# Graph configuration
plt.title("Comparison of Experimental vs. Theoretical Centroids")

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('off')
plt.show()

