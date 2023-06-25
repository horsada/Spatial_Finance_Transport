import os
import numpy as np
import osgeo.gdal as gdal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
import math
import pandas as pd

def get_tif_files(directory_path):
    """
    Returns a list of all files ending in .tif in the given directory path.

    Args:
        directory_path (str): Path of the directory to search for .tif files.

    Returns:
        list: List of all files ending in .tif in the given directory path.
    """
    # Initialize an empty list to store the .tif files
    tif_files = []

    # Loop through all files in the directory
    for file in os.listdir(directory_path):
        # Check if the file ends with .tif
        if file.endswith('.tif'):
            # Append the file path to the list
            tif_files.append(os.path.join(directory_path, file))

    # Return the list of .tif files
    return tif_files

def tif_to_array(tif_path):
    ds = gdal.Open(tif_path)
    array = np.array([ds.GetRasterBand(i).ReadAsArray() for i in range(1, ds.RasterCount + 1)])
    array = array.transpose((1,2,0))

    print("array shape: {}".format(array.shape))
    return array



def apply_pca_to_wv2_image(ms_img, n_components=5):
    """
    Apply PCA to resampled multispectral bands of WV2 images for change detection.
    
    Args:
        ms_img (ndarray): Multispectral image (MS) of WV2.
        n_components (int): Number of PCA components to retain (default: 5).
    
    Returns:
        ndarray: Reconstructed images using the first n_components of PCA.
    """
    # Normalize the pixel values
    images = ms_img.astype(float)
    images /= 255.0  # assuming pixel values range from 0 to 255

    # Transpose the array to change the shape
    images = np.transpose(images, (2, 0, 1))

    print("images shape after stacking: {}".format(images.shape))
    n_samples, width, height = images.shape
    
    # Reshape the images to a 2D array
    X = np.reshape(images, (n_samples, -1))
    print("X shape: {}".format(X.shape))

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Apply PCA with n_components
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Reconstruct the images using the first n_components of PCA
    pc1_to_pcN = pca.components_[:n_components]
    reconstructed_images = np.dot(pca.transform(X)[:,:n_components], pc1_to_pcN) + pca.mean_
    reconstructed_images = np.reshape(reconstructed_images, (n_samples, width, height))
    print("reconstructed images shape: {}".format(reconstructed_images.shape))
    
    return reconstructed_images


def extract_vehicle_centroids(pc_change_image, channel_index=3, aspect_ratio_threshold=0.001, rectangularity_threshold=0.001, area_threshold=50):
    """
    Extract centroids of moving vehicles from PCA change image based on morphological features.

    Args:
    pc_change_image (ndarray): PCA change image.
    channel_index (int): Index of the channel to use for processing (default: 0).
    aspect_ratio_threshold (float): Threshold for aspect ratio (default: 0.5).
    rectangularity_threshold (float): Threshold for rectangularity (default: 0.5).
    area_threshold (int): Threshold for area (default: 100).

    Returns:
    ndarray: Centroids of moving vehicles.
    """
    # Extract the specific channel for processing
    pc_change_image = pc_change_image[3, :, :]

    # Convert PCA change image to binary image based on thresholding
    _, binary_image = cv2.threshold(pc_change_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list to store centroids
    centroids = []

    # Loop through contours and extract centroids of objects based on morphological features
    for contour in contours:
        # Fit minimum area bounding rectangle to contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

    # Calculate width and height of bounding rectangle
    width = rect[1][0]
    height = rect[1][1]

    # Calculate aspect ratio and rectangularity of contour
    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
    rectangularity = cv2.contourArea(contour) / (width * height) if width * height > 0 else 0

    # Check if aspect ratio and rectangularity meet threshold criteria
    if aspect_ratio >= aspect_ratio_threshold and rectangularity >= rectangularity_threshold:
        # Extract centroid of contour and add to list
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))

    # Convert centroids to numpy array and return
    centroids = np.array(centroids)

    return centroids


def pair_centroids(centroids, threshold, time_lag=0.13):
    time_lag = time_lag * 0.000277778
    # Calculate the distance between each pair of centroids
    distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            x1, y1 = centroids[i]
            x2, y2 = centroids[j]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append((distance, i, j))
    
    # Sort the distances in ascending order
    distances.sort()
    
    # Pair the nearest centroids if their distance is below the threshold
    pairs = []
    paired = set()
    for distance, i, j in distances:
        if distance <= threshold and i not in paired and j not in paired:
            pairs.append((centroids[i], centroids[j]))
            paired.add(i)
            paired.add(j)

    # Calculate distance and speed for each pair
    results = []
    speeds = []
    for pair in pairs:
        x1 = pair[0][0]
        y1 = pair[0][1]
        x2 = pair[1][0]
        y2 = pair[1][1]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distance = distance * 5e-4 # convert to km 
        speed = distance / time_lag
        speeds.append(speed)
        results.append({'pair': pair, 'distance': distance, 'speed': speed})

    if len(pairs) > 0:
        avg_speed = np.median(speeds)
        avg_speed = avg_speed * 0.621371
    else:
        avg_speed = 0
    
    return pairs, avg_speed


################################################################################################ 

def speed_estimation(IMAGE_DIR, SPEED_ESTIMATION_DIR, SITE_NAME):
    ms_image_paths = get_tif_files(IMAGE_DIR)

    ms_images_list = []

    for ms_image_path in ms_image_paths:

        ms_img = tif_to_array(ms_image_path)

        ms_images_list.append(ms_img)


    recon_imgs_list = []

    for ms_img in ms_images_list:

        recon_imgs = apply_pca_to_wv2_image(ms_img, n_components=4)

        recon_imgs_list.append(recon_imgs)


    vehicle_centroid_list = []

    for recon_img in recon_imgs_list:

        centroids = extract_vehicle_centroids(recon_img)

        vehicle_centroid_list.append(centroids)


    results_list = []
    avg_speeds_list = []

    for centroids, recon_img in zip(vehicle_centroid_list, recon_imgs_list):

        results, avg_speed = pair_centroids(centroids, threshold=9.5, time_lag=0.13)

        results_list.append(results)

        avg_speeds_list.append(avg_speed)

    # saving data
    df_avg_speed = pd.DataFrame(columns=['image_id', 'avg_speed_estimate'])

    df_avg_speed = df_avg_speed.append({'image_id': SITE_NAME, 'avg_speed_estimate': avg_speed}, ignore_index=True)

    df_avg_speed.to_csv(SPEED_ESTIMATION_DIR+'avg_speed_estimates.csv')

    return True, avg_speeds_list[0]