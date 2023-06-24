from pipeline.trafficapi import trafficapi
from pipeline.inference import inference
from pipeline.image_preprocessing import convert_ms_images_to_rgb
from pipeline.speedEstimation import speed_esimation
from pipeline.input import save_args
from pipeline.aadtpreprocessing import aadt_preprocess

import argparse
import os

ROOT_DIR_PATH = os.path.abspath('../Spatial_Finance_Transport/')
IMAGE_DIR = os.path.join(ROOT_DIR_PATH, 'data/satellite_images/')
PROCESSED_IMAGE_DIR = os.path.join(ROOT_DIR_PATH, 'data/satellite_images/processed/')
SPEED_ESTIMATION_DIR = os.path.join(ROOT_DIR_PATH, 'data/predicted/speed_data/')
TIME_DATE_DIR = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/time_data/')
LINK_LENGTH_DIR = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/link_length_data/')
AADT_DIR = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/aadt/')
MODEL_PATH = os.path.join(ROOT_DIR_PATH, 'models/object_detection_models/best.pt')
VEHILCE_COUNT_DIR = os.path.join(ROOT_DIR_PATH, 'data/predicted/vehicle_counts/')
AADT_PROCESSED_DIR = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/aadt/processed/')

def main(**kwargs):
    # Access the keyword arguments
    SITE_NAME = kwargs.get('arg1')
    YEAR = kwargs.get('arg2')
    LINK_LENGTH = kwargs.get('arg3')
    TIME_DATE = kwargs.get('arg4')

    # print keyword arguments
    print("Site name:", SITE_NAME)
    print("Year:", YEAR)
    print("Link length:", LINK_LENGTH)
    print("Time and date:", TIME_DATE)

    print("------ Saving arguments to folders --------")
    saves_args_successful = save_args(SITE_NAME, LINK_LENGTH, LINK_LENGTH_DIR, TIME_DATE, TIME_DATE_DIR)

    if saves_args_successful:
        print("--------------- Successfully Saved Keyword arguments -----------------")


    print("------- Performing Traffic API Data Preparation -----------")

    trafficapi_success = trafficapi(SITE_NAME, YEAR, AADT_DIR)

    if trafficapi_success:
        print("--------------- Successfully Performed Traffic API Data Preparation -----------------")

    print("----------- Pre-processing satellite images --------------")

    image_pre_process_success = convert_ms_images_to_rgb(IMAGE_DIR, PROCESSED_IMAGE_DIR)

    if image_pre_process_success:
        print("--------------- Successfully Performed Satellite Image Pre-Processing -----------------")

    print("--------- Performing inference -------------")

    inference_success, n_vehicles = inference(PROCESSED_IMAGE_DIR, visualize=False, model_path=MODEL_PATH, vehicle_counts_path=VEHILCE_COUNT_DIR)

    if inference_success:
        print("--------------- Successfully Performed Object Detection Inference -----------------")
        print("Number of vehicles detected: {}".format(n_vehicles))

    print("-------- Performing Average Vehicle Speed --------------------")

    speed_esimation_success, speed_estimate = speed_esimation(IMAGE_DIR, SPEED_ESTIMATION_DIR, SITE_NAME)

    if speed_esimation_success:
        print("--------------- Successfully Performed Speed Estimation -----------------")
        print("Average vehicle speed: {}".format(speed_estimate))

    aadt_preprocess_success = aadt_preprocess(AADT_DIR, AADT_PROCESSED_DIR)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Define the keyword arguments
    parser.add_argument('--arg1', type=str, help='Site name')
    parser.add_argument('--arg2', type=str, help='Year: YYYY')
    parser.add_argument('--arg3', type=float, help='Link length (km)')
    parser.add_argument('--arg4', type=str, help='Acquisition Time: hh:1-31/1-12')


    args = parser.parse_args()

    # Pass the keyword arguments to the main function
    main(arg1=args.arg1, arg2=args.arg2, arg3=args.arg3, arg4=args.arg4)
