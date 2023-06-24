from sahi.utils.yolov5 import (
    download_yolov5s6_model,
)
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
from skimage import io
from skimage import exposure, transform
import random
import os
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
import os
import rasterio
import numpy as np
from skimage.transform import resize
import csv

"""## Helper Functions"""

def extract_substring(string):
    """
    Extract a substring from a string based on characters after the last slash ("/") and before the dot (".").

    Args:
        string (str): Input string.

    Returns:
        str: Extracted substring.
    """
    # Find the index of the last slash and the dot
    slash_index = string.rfind("/")
    dot_index = string.find(".")

    # Extract the substring based on the last slash and dot indices
    if slash_index != -1 and dot_index != -1:
        substring = string[slash_index + 1:dot_index]
    else:
        substring = ""

    return substring



def save_results_to_csv(results, file_path, image_id):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Open a CSV file for writing
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(['image_id', 'x_min', 'x_max', 'y_min', 'y_max', 'category_name', 'area'])

        # Write each result as a row in the CSV file

        if len(results) == 0:
          # Check if bbox is empty
            writer.writerow([image_id, 0, 10, 0, 10, 'Small Car', 100])

        else:
          for result in results:
              # Extract the relevant values from the result dictionary
              bbox = result['bbox']
              category_name = result['category_name']
              area = result['area']

              # Write the row to the CSV file
              writer.writerow([image_id, bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3], category_name, area])

    print(f'Saved results to CSV file: {file_path}')



def get_files_in_directory(directory):
    """
    Get a list of all files in a directory.

    Args:
        directory (str): Directory path.

    Returns:
        list: List of files in the directory.
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            files.append(file_path)
    return files


"""## Sliced Inference YOLOv5

- To perform sliced prediction we need to specify slice parameters. In this example we will perform prediction over slices of 512x512 with an overlap ratio of 0.25:

Need to fix Hounslow M4 2188A, Trafford M60 9083A
"""

def inference(processed_image_dir,  model_path, vehicle_counts_path, visualize=False):

    processed_image_paths = get_files_in_directory(processed_image_dir)

    # instantiate model
    detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model_path=model_path,
    confidence_threshold=0.005,
    device="cpu", # or 'cuda:0'
    )

    for processed_image_path in processed_image_paths:

        image_id = extract_substring(processed_image_path)

        result = get_sliced_prediction(
            processed_image_path,
            detection_model,
            slice_height = 256,
            slice_width = 256,
            overlap_height_ratio = 0.25,
            overlap_width_ratio = 0.25,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.5,
            postprocess_class_agnostic=False,
        )

        result_coco = result.to_coco_predictions(image_id=processed_image_path)

        print("Number of vehicles detected: {}".format(len(result_coco)))

        if visualize:
            result.export_visuals(export_dir="result_data/", file_name=image_id)

        save_results_to_csv(result_coco, vehicle_counts_path+'vehicle_counts_'+image_id+'.csv', image_id=image_id)
    
    return True, len(result_coco)