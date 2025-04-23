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
    Analyzes and filters contours based on their area, drawing them with additional information.
    
    Parameters:
    - image: Input grayscale image
    - contours: List of contours obtained from cv2.findContours
    - area_min: Minimum area to filter contours (default: 70000)
    - area_max: Maximum area to filter contours (default: 400000)
    - debug: If True, returns additional debugging information
    
    Returns:
    - image_areas: Image with numbered contours and centroids
    - filtered_image: Image with filtered contours colored
    - dict: (optional) Additional information if debug=True

    """
    
    # Calculate and sort areas of all contours
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    contour_areas.sort(reverse=True)

    # Create copies of the image for visualization
    image_areas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    filtered_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Lists to store results
    filtered_areas = []
    centroids = []

    
    # Process each contour
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter by minimum and maximum area
        if area_min <= area <= area_max:
            filtered_areas.append((i + 1, area))
            
            # Generate random color
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # Draw filtered contour
            cv2.drawContours(filtered_image, [contour], -1, color, thickness=5)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append([cx, cy])
                
                # Draw number and centroid on image_areas
                cv2.putText(image_areas, f"{i+1}", (cx, cy), 
                          cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 3)
                cv2.circle(image_areas, (cx, cy), 15, (255, 255, 255), -1)

    # Print information
    print("Areas sorted from largest to smallest:", contour_areas)
    print(f"Minimum area: {area_min}")
    print(f"Maximum area: {area_max}")
    print("Contours Filtered by Area")
    for num, area in filtered_areas:
        print(f"Contour {num}, Area = {area:.2f}")
    
    if debug:
        debug_info = {
            'total_areas': contour_areas,
            'filtered_areas': filtered_areas,
            'centroids': centroids
        }
        return image_areas, filtered_image, debug_info
    
    return image_areas, filtered_image, centroids


def filtrar_y_recortar_circulos(image, contours, centroids, output_folder, tolerance=0.20, circularity_min=0.75, 
                            circularity_max=1.1, expansion_factor=0.15):
    """
    Filters approximately circular contours, calculates average circles, and crops regions of interest.

    Parameters:
    - image: Input image in BGR format.
    - contours: List of contours obtained with cv2.findContours.
    - centroids: List of [cx, cy] coordinates of the contour centroids.
    - output_folder: Directory where the cropped images will be saved.
    - tolerance: Percentage tolerance for the area range (default: 0.20).
    - circularity_min: Minimum circularity threshold (default: 0.75).
    - circularity_max: Maximum circularity threshold (default: 1.1).
    - expansion_factor: Factor to expand the crop beyond the radius (default: 0.15).

    Returns:
    - imagep: Original image with rectangles drawn around the cropped regions.
    - circles: Image with the average circles drawn.
    - cropped_images: List of cropped images.
    - mask: Mask with the minimum circles enclosing the filtered contours.

    """

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Calculate the maximum area of the contours
    if not contours:
        print("No contours found.")
        return image, np.zeros_like(gray_image), [], np.zeros_like(gray_image)
    max_area = max(cv2.contourArea(contour) for contour in contours)
    print("Maximum area:", max_area)

    # Define the tolerance range (default 20% of the maximum area)
    min_area = max_area * (1 - tolerance)
    max_area_tolerance = max_area * (1 + tolerance)
    print("Minimum area:", min_area)
    print("Maximum area with tolerance:", max_area_tolerance)

    # Create a black mask
    mask = np.zeros_like(gray_image)

    # Filter contours and calculate radii
    filtered_contours = []
    radii = []  # List to store the radii of the circles

    for contour in contours:
        contour_area = cv2.contourArea(contour)
    
        # Filter by area within the tolerance range
        if min_area <= contour_area <= max_area_tolerance:
            # Get the minimum enclosing circle for the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
        
            # Check if the shape is approximately circular
            circularity = contour_area / circle_area  # Should be close to 1 for a circle
        
            if circularity_min <= circularity <= circularity_max:
                filtered_contours.append(contour)
                radii.append(radius)
                cv2.circle(mask, (int(x), int(y)), int(radius), (255), thickness=3)

    # Calculate the average radius
    average_radius = int(np.mean(radii)) if radii else 0
    print("Average radius:", average_radius)

    # Draw average circles at the centroids
    circles = np.zeros_like(gray_image)
    for cx, cy in centroids:
        cv2.circle(circles, (cx, cy), average_radius, (255, 0, 0), thickness=3)  # Blue for average circles

    # List to store the cropped images
    cropped_images = []
    imagep = image.copy()  # Copy of the original image to draw rectangles
    expansion = int(average_radius * expansion_factor)  # Additional margin for cropping

    # Crop each average circle
    for i, (cx, cy) in enumerate(centroids):
        # Define the crop boundaries
        x_min = max(0, cx - average_radius - expansion)
        x_max = min(imagep.shape[1], cx + average_radius + expansion)
        y_min = max(0, cy - average_radius - expansion)
        y_max = min(imagep.shape[0], cy + average_radius + expansion)

        # Crop the region of the image
        cropped = image[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped)

        # Save the cropped image
        output_path = os.path.join(output_folder, f"filtered_shape_{i+1}.jpg")
        os.makedirs(output_folder, exist_ok=True)  # Create directory if it doesn't exist
        cv2.imwrite(output_path, cropped)

        # Draw a rectangle around the cropped area on the original image
        cv2.rectangle(imagep, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=4)

    return imagep, circles, cropped_images, mask

################################---------------##############################

def recortar_circulo(image, clahe_clip=2.0, clahe_tile=16):
    """
    Detects and crops the largest circle in the image using HoughCircles.
    
    Parameters:
    - image: Input image in BGR format.
    - clahe_clip: Contrast limit for CLAHE (default: 1.0).
    - clahe_tile: Grid size for CLAHE (default: 4).
    
    Returns:
    - output_image: Original image with the detected circle drawn.
    - centered_image: Image with the cropped circle centered on a black background.
    - radius: Radius of the detected circle.
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
        
    # If no circles are detected, return default or None values
    return None, 0, 0, 0

def  procesar_contornos_kmeans(image, threshold_value=100, area_min=100, area_max=5000, kernel_size=3, clahe_clip=1.0, clahe_tile=4, radius=211, x=255, y=255):
    """
    Processes the image to find contours, filter them by area, and apply K-Means to the centroids.
    
    Parameters:
    - image: Input image in BGR format.
    - threshold_value: Threshold value for binarization (default: 100).
    - area_min: Minimum area to filter contours (default: 100).
    - area_max: Maximum area to filter contours (default: 5000).
    - kernel_size: Kernel size for morphological operations (default: 3).
    - clahe_clip: Contrast limit for CLAHE (default: 1.0).
    - clahe_tile: Grid size for CLAHE (default: 4).
    
    Returns:
    - contours_image: Image with all contours drawn.
    - filtered_image: Image with contours filtered by area.
    - kmeans_image: Image with centroids grouped by K-Means.
    """
    # Get the size of the image
    height, width = image.shape[:2]  # Extract height and width (ignore channels if any)
    image_size = (width, height)  # Return as (width, height)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    img_clahe = clahe.apply(gray)
    
    smoothed_image = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    bilateral_image = cv2.bilateralFilter(smoothed_image, 7, 7, 75)
    _, binary = cv2.threshold(bilateral_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=1)
    closed_edges = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    clean_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(clean_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), thickness=1)
    
    filtered_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    centroids = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area_min <= area <= area_max:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.drawContours(filtered_image, [contour], -1, color, thickness=1)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append([cx, cy])
                cv2.putText(filtered_image, f"{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    kmeans_image = image.copy()
    kmeans_centers = []
    mask = np.zeros_like(image, dtype=np.uint8)
    #cv2.circle(mask, (x, y), radius, (255, 255, 255), 2)  # Draw the circle on the mask
    if len(centroids) > 1:
        k = min(500, len(centroids))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(np.array(centroids, dtype=np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Draw the detected centers on the image
        for center in centers:
            cx, cy = int(center[0]), int(center[1])
            kmeans_centers.append([cx, cy])  # Save the coordinates
            cv2.circle(kmeans_image, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(mask, (cx, cy), 5, (255, 255, 255), -1)
        
    
    return kmeans_image, mask, kmeans_centers, image_size, contours_image, filtered_image
