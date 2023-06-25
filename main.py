from pipeline.trafficapi import trafficapi
from pipeline.inference import inference
from pipeline.image_preprocessing import convert_ms_images_to_rgb
from pipeline.speedEstimation import speed_estimation
from pipeline.input import save_args
from pipeline.potentialSites import potential_sites
from pipeline.aadtPreprocessing import aadt_preprocess
from pipeline.aadtTraining import aadt_training
from pipeline.aadtImplementation import aadt_implementation
from pipeline.ghgImplementation import ghg_implementation


import argparse
import os

ROOT_DIR_PATH = os.path.abspath('../Spatial_Finance_Transport/')
IMAGE_DIR = os.path.join(ROOT_DIR_PATH, 'data/satellite_images/')
PROCESSED_IMAGE_DIR = os.path.join(ROOT_DIR_PATH, 'data/satellite_images/processed/')
SPEED_ESTIMATION_DIR = os.path.join(ROOT_DIR_PATH, 'data/predicted/speed_data/')
TIME_DATE_DIR = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/time_data/')
LINK_LENGTH_DIR = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/link_length_data/')
AADT_DIR = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/aadt/')
MODEL_PATH = os.path.join(ROOT_DIR_PATH, 'models/object_detection_models/yolov5_road_vehicles.pt')
VEHILCE_COUNT_DIR = os.path.join(ROOT_DIR_PATH, 'data/predicted/vehicle_counts/')
AADT_PROCESSED_DIR = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/aadt/processed/')
GHG_EMISSIONS_DATA_PATH = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/ghg_emissions/')
GHG_PROCESSED_PATH = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/ghg_emissions/GHG_potential_sites.csv')
NN_MODEL_PATH = os.path.join(ROOT_DIR_PATH, "models/aadt_models/")
VEHICLE_COUNTS_PATH = os.path.join(ROOT_DIR_PATH, 'data/predicted/vehicle_counts/')
TRANSFORM_PATH = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/aadt/processed/')
TRAFFIC_COUNTS_PATH = os.path.join(ROOT_DIR_PATH, 'data/predicted/traffic_counts/')
AADT_PRED_PATH = os.path.join(ROOT_DIR_PATH, 'data/predicted/aadt/')
GHG_EMISSIONS_PRED_PATH = os.path.join(ROOT_DIR_PATH, 'data/predicted/ghg_emissions/')
TRUE_SPEED_DIR = os.path.join(ROOT_DIR_PATH, 'data/ground_truth_data/speed_data/')

def main(**kwargs):
    # Access the keyword arguments
    LA = kwargs.get('arg0')
    SITE_NAME = kwargs.get('arg1')
    YEAR = kwargs.get('arg2')
    LINK_LENGTH = kwargs.get('arg3')
    TIME_DATE = kwargs.get('arg4')
    TRUE_SPEED = kwargs.get('arg5')
    SAT_IMAGE_PATH = kwargs.get('arg6')

    # print keyword arguments
    print("LA:", LA)
    print("Site name:", SITE_NAME)
    print("Year:", YEAR)
    print("Link length:", LINK_LENGTH)
    print("Time and date:", TIME_DATE)
    print("True speed:", TRUE_SPEED)
    print("Satellite image path:", SAT_IMAGE_PATH)

    print("------ Saving arguments to folders --------")
    saves_args_successful = save_args(LA, SITE_NAME, LINK_LENGTH, LINK_LENGTH_DIR, TRUE_SPEED, TRUE_SPEED_DIR, TIME_DATE, TIME_DATE_DIR, SAT_IMAGE_PATH, IMAGE_DIR)

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

    speed_esimation_success, speed_estimate = speed_estimation(IMAGE_DIR, SPEED_ESTIMATION_DIR, LA, SITE_NAME)

    if speed_esimation_success:
        print("--------------- Successfully Performed Speed Estimation -----------------")
        print("Average vehicle speed: {}".format(speed_estimate))

    print("-------- Performing Potential Sites Processing --------------------")

    potential_sites_success = potential_sites(CHOSEN_COUNT_SITES=[(LA, SITE_NAME)], AADT_DATA_PATH=AADT_DIR, GHG_PROCESSED_PATH=GHG_PROCESSED_PATH, GHG_EMISSIONS_DATA_PATH=GHG_EMISSIONS_DATA_PATH)

    if potential_sites_success:
        print("--------------- Successfully Performed Potential Sites Processing -----------------")


    print("-------- Performing AADT Pre-Processing --------------------")

    aadt_preprocess_success = aadt_preprocess(aadt_path=AADT_DIR, processed_path=AADT_PROCESSED_DIR)

    if aadt_preprocess_success:
        print("--------------- Successfully Performed AADT Pre-Processing -----------------")


    print("-------- Performing ANN AADT Training --------------------")

    aadt_training_success = aadt_training(AADT_PROCESSED_PATH=AADT_PROCESSED_DIR, NN_MODEL_PATH=NN_MODEL_PATH)

    if aadt_training_success:
        print("--------------- Successfully Performed AADT ANN Training -----------------")

    print("-------- Performing AADT Implementation --------------------")

    aadt_implementation_success = aadt_implementation(MODELS_PATH=NN_MODEL_PATH, VEHICLE_COUNTS_PATH=VEHICLE_COUNTS_PATH, 
                                                      TRUE_SPEED_PATH=TRUE_SPEED_DIR, 
                            TIME_PATH=TIME_DATE_DIR, LINK_LENGTH_PATH=LINK_LENGTH_DIR, TRAFFIC_COUNTS_PATH=TRAFFIC_COUNTS_PATH, 
                            TRANSFORM_PATH=TRANSFORM_PATH, PRED_SPEED_PATH=SPEED_ESTIMATION_DIR, AADT_PRED_PATH=AADT_PRED_PATH)
    
    if aadt_implementation_success:
        print("--------------- Successfully Performed AADT Implementation -----------------")

    ghg_implementation_success = ghg_implementation(AADT_PRED_PATH, VEHILCE_COUNT_DIR, GHG_EMISSIONS_PRED_PATH)

    if ghg_implementation_success:
        print("--------------- Successfully Performed GHG Implementation -----------------")

if __name__ == '__main__':

    print("-"*50)
    print("Starting Program...")
    parser = argparse.ArgumentParser()

    # Define the keyword arguments
    parser.add_argument('--arg0', type=str, help='LA')
    parser.add_argument('--arg1', type=str, help='Site name')
    parser.add_argument('--arg2', type=str, help='Year: YYYY')
    parser.add_argument('--arg3', type=float, help='Link length (km)')
    parser.add_argument('--arg4', type=str, help='Acquisition Time: hh:1-31/1-12')
    parser.add_argument('--arg5', type=int, help='Speed')
    parser.add_argument('--arg6', type=str, help='Sat Image Path')


    args = parser.parse_args()

    # Pass the keyword arguments to the main function
    main(arg0=args.arg0, arg1=args.arg1, arg2=args.arg2, arg3=args.arg3, arg4=args.arg4, arg5=args.arg5, arg6=args.arg6)

    print("-"*50)
    print("Exiting Program...")
