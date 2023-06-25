import pandas as pd
import numpy as np
import os
import torch
import random
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import albumentations as album
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import torchmetrics
from torchmetrics import MeanAbsolutePercentageError
from glob import glob
import plotly.graph_objs as go

# values in miles
BLACKBURN_LINK_LENGTH = 0.94
HOUNSLOW_LINK_LENGTH = 1.07
HAVERING_LINK_LENGTH = 3.79
TRAFFORD_LINK_LENGTH = 0.65
LUTON_LINK_LENGTH = 0.85

# GSD
GSD = 50

NORMALISE_DICT = {
    'Total_N15': 'total_volume',
    'Small_N15': '0-520cm',
    'Medium_N15': '521-660cm',
    'Large_N15': '661-1160cm',
    'Very Large_N15': '1160+cm'
}

COLUMN_NAMES = ['aadt', 'cars_and_taxis', 'buses_and_coaches', 'lgvs', 'all_hgvs']

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


def categorize_bbox_size(df):
    """
    Categorize the maximum value of (x_max - x_min) and (y_max - y_min) into categories and count occurrences.

    Args:
        df (pd.DataFrame): DataFrame containing the bounding box data.

    Returns:
        list: List of tuples containing the category label and count.
    """
    # Calculate the maximum of (x_max - x_min) and (y_max - y_min) for each row
    df['max_size'] = df[['x_max', 'x_min', 'y_max', 'y_min']].apply(lambda x: max(x[0] - x[1], x[2] - x[3]), axis=1)

    # Define the category labels and corresponding size ranges
    categories = {
        'Small': (0, 520),
        'Medium': (520, 660),
        'Large': (661, 1160),
        'Very Large': (1161, float('inf'))
    }

    # Initialize a dictionary to store counts for each category
    counts = {category: 0 for category in categories}

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the max size value for the row
        max_size = row['max_size'] * GSD

        # Categorize the max size value and update the counts
        for category, size_range in categories.items():
            if size_range[0] <= max_size < size_range[1]:
                counts[category] += 1

    counts_df = pd.DataFrame([counts])

    # Add a column for the sum of all counts
    counts_df.insert(0, 'Total', counts_df.sum(axis=1))
    return counts_df


def save_float_to_csv(float_values, column_names, image_id, file_name):
    """
    Save float values to a CSV file with the specified column names and file name.

    Args:
        float_values (List[float]): The list of float values to be saved.
        column_names (List[str]): The list of column names in the CSV file.
        image_id (str): The image ID associated with the float values.
        file_name (str): The name of the CSV file to be saved.
    """
    # Create a dictionary of column names and corresponding float values
    data = {'image_id': image_id}
    for name, value in zip(column_names, float_values):
        data[name] = [value]

    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(file_name, index=False)


def calculate_N15(df_v, df_N, df_l, MODE='none'):
    # Merge the three input dataframes on the 'image_id' column

    cols = ['Total', 'Small',	'Medium',	'Large',	'Very Large']
    avg_mph = df_v.iloc[0]['avg_mph']
    link_length = df_l.iloc[0]['link_length']

    print("Calculating traffic counts in MODE: {}".format(MODE))

    for col in cols:

      # Calculate N15 using the formula
      df_N[col+'_N15'] = 0.25 * avg_mph * df_N[col] / link_length

      if MODE == '50_traffic_counts':
        df_N[col+'_N15'] = df_N[col+'_N15'] * 0.5

      if MODE =='150_traffic_counts':
        df_N[col+'_N15'] = df_N[col+'_N15'] * 1.5

    # Return a dataframe with only the 'image_id' and 'N15' columns
    return df_N


def get_files_by_prefix(directory, prefix):
    """
    Returns a list of file paths in a directory that match the start of a string.

    Args:
    directory (str): the path to the directory to search in.
    prefix (str): the prefix of the file names to match.

    Returns:
    A list of file paths that match the specified prefix.
    """
    matching_files = []
    for filename in os.listdir(directory):
        if prefix in filename:
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                matching_files.append(file_path)
    return matching_files


class NeuralNetwork(nn.Module):
    def __init__(self, name):
        super(NeuralNetwork, self).__init__()

        self.name = name
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 7),
            nn.Linear(7,7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(7,5),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

def get_model_list(MODELS_PATH):
    model_paths = get_files_by_prefix(MODELS_PATH, prefix='nn_model')

    model_list = []

    for model_path in model_paths:

        name = extract_substring(model_path)

        name = name.replace('nn_model_', '')

        model = NeuralNetwork(name=name)

        model.load_state_dict(torch.load(model_path))

        model.eval()

        print("model name: {}".format(model.name))

        model_list.append(model)

    return model_list

def load_and_process_vehicle_counts(VEHICLE_COUNTS_PATH):

    df_vehicle_count_list = []

    vehicle_count_paths = get_files_in_directory(VEHICLE_COUNTS_PATH)

    for vehicle_count_path in vehicle_count_paths:
        df = pd.read_csv(vehicle_count_path)

        df_vehicle_count_list.append(df)

    df_processed_vehicle_counts_list = []

    for df in df_vehicle_count_list:
        df_processed_vehicle_count = categorize_bbox_size(df)

        df_processed_vehicle_count['image_id'] = df['image_id'].astype(str)

        print(df_processed_vehicle_count.iloc[0]['image_id'])

        df_processed_vehicle_counts_list.append(df_processed_vehicle_count)

    #print("df_processed_vehicle_counts_list: {}".format(df_processed_vehicle_counts_list))

    return df_processed_vehicle_counts_list


def load_true_speed(TRUE_SPEED_PATH):
    true_speed_paths = get_files_in_directory(TRUE_SPEED_PATH)

    df_speed_list = []

    if_true_speed = True

    for true_speed_path in true_speed_paths:

        df = pd.read_csv(true_speed_path, skipinitialspace=True)

        df['image_id'] = df['image_id'].astype(str)

        print("df['avg_mph'].values: {}".format(df['avg_mph'].values))

        if -1 in df['avg_mph'].values:
            print("Found -1 in avg_mph column of DataFrame:", true_speed_path)
            if_true_speed = False

        df_speed_list.append(df)
    
    return df_speed_list, if_true_speed



def load_link_lengths(LINK_LENGTH_PATH):

    link_length_paths = get_files_in_directory(LINK_LENGTH_PATH)

    df_link_length_list = []

    for link_length_path in link_length_paths:

        df = pd.read_csv(link_length_path, skipinitialspace=True)

        df['image_id'] = df['image_id'].astype(str)

        df_link_length_list.append(df)
    
    return df_link_length_list

def traffic_counts(df_processed_vehicle_counts_list, df_speed_list, 
                   df_link_length_list, TRAFFIC_COUNTS_PATH, 
                   df_speed_estimate=None, TRUE_SPEED=False, MODE='none'):

    if TRUE_SPEED:

        for df_processed_vehicle_counts in df_processed_vehicle_counts_list:

            for df_speed in df_speed_list:

                for df_link_length in df_link_length_list:

                        if ( df_processed_vehicle_counts.iloc[0]['image_id'] == df_speed.iloc[0]['image_id'] ) and ( df_processed_vehicle_counts.iloc[0]['image_id']  == df_link_length.iloc[0]['image_id'] ):

                            print("found match for: {}".format(df_link_length.iloc[0]['image_id']))

                            image_id = df_link_length.iloc[0]['image_id']

                            df_traffic_count = calculate_N15(df_speed, df_processed_vehicle_counts, df_link_length, MODE=MODE)

                            df_traffic_count.to_csv(TRAFFIC_COUNTS_PATH+'traffic_count_'+image_id+'.csv')

    else:
        for df_processed_vehicle_counts in df_processed_vehicle_counts_list:

            print("Calculating traffic counts...")

            print(df_processed_vehicle_counts)

            for df_link_length in df_link_length_list:

                if ( df_processed_vehicle_counts.iloc[0]['image_id']  == df_link_length.iloc[0]['image_id'] ):

                    print("found match for: {}".format(df_link_length.iloc[0]['image_id']))

                    df_speed = df_speed_estimate.loc[df_speed_estimate['image_id'] == df_processed_vehicle_counts.iloc[0]['image_id'],
                                                    ['image_id', 'avg_mph']]

                    df_speed = df_speed.rename(columns={'avg_speed_estimate': 'avg_mph'})

                    image_id = df_link_length.iloc[0]['image_id']

                    df_traffic_count = calculate_N15(df_speed, df_processed_vehicle_counts, df_link_length, MODE=MODE)

                    df_traffic_count.to_csv(TRAFFIC_COUNTS_PATH+'traffic_count_'+image_id+'.csv')


def load_transform(TRANSFORM_PATH):

    transform_prefix = 'transform'

    transform_paths = get_files_by_prefix(TRANSFORM_PATH, transform_prefix)

    df_transform_list = []

    for transform_path in transform_paths:

        df = pd.read_csv(transform_path)

        df = df.set_index('Unnamed: 0')

        df.name = extract_substring(transform_path).lower()

        df_transform_list.append(df)

    return df_transform_list


def transform_counts(df_transform_list, df_processed_vehicle_counts_list):

    print("Transforming counts...")

    transform_cols = ['Total_N15', 'Small_N15', 'Medium_N15', 'Large_N15', 'Very Large_N15']

    for df_transform in df_transform_list:

        for df_processed_vehicle_counts in df_processed_vehicle_counts_list:

            print(df_processed_vehicle_counts.head())

            if df_transform.name[-5:] in df_processed_vehicle_counts.iloc[0]['image_id']:

                print("found a match for: {}".format(df_processed_vehicle_counts.iloc[0]['image_id']))

                for transform_col in transform_cols:

                    min_val, max_val = df_transform.loc['min', NORMALISE_DICT[transform_col]], df_transform.loc['max', NORMALISE_DICT[transform_col]]

                    df_processed_vehicle_counts.loc[:, transform_col] = (df_processed_vehicle_counts[transform_col] - min_val) / (max_val - min_val)

    return df_processed_vehicle_counts_list


def load_time(TIME_PATH):
  
    time_paths = get_files_in_directory(TIME_PATH)

    df_time_list = []

    for time_path in time_paths:

        df = pd.read_csv(time_path)

        df_time_list.append(df)

    return df_time_list


def concatenate_inputs(df_processed_vehicle_counts_list, df_speed_list, df_time_list, if_true_speed=False):

    df_aadt_features_list = []

    print("Concatenating inputs...")

    for df_processed_vehicle_counts in df_processed_vehicle_counts_list:

        for df_speed in df_speed_list:

            for df_time in df_time_list:

                if ( df_processed_vehicle_counts.iloc[0]['image_id'] == df_speed.iloc[0]['image_id'] ) and ( df_processed_vehicle_counts.iloc[0]['image_id']  == df_time.iloc[0]['image_id'] ):

                    print("Found match for: {}".format(df_processed_vehicle_counts.iloc[0]['image_id']))

                    print("if_true_speed: {}".format(if_true_speed))

                    if if_true_speed: 

                        df = pd.concat([df_processed_vehicle_counts[['image_id', 'Total_N15',	'Small_N15', 'Medium_N15', 'Large_N15', 'Very Large_N15']], df_speed[['avg_mph']], df_time[['day', 'month', 'hour']]], axis=1)

                    else:

                        df = pd.concat([df_processed_vehicle_counts[['image_id', 'Total_N15',	'Small_N15', 'Medium_N15', 'Large_N15', 'Very Large_N15']], df_speed[['avg_mph']], df_time[['day', 'month', 'hour']]], axis=1)

                    df_aadt_features_list.append(df)

    return df_aadt_features_list


def load_speed_estimate(PRED_SPEED_PATH):

    df_speed_estimate = pd.read_csv(PRED_SPEED_PATH+'avg_speed_estimates.csv', skipinitialspace=True)

    return df_speed_estimate

def aadt_implementation(MODELS_PATH, VEHICLE_COUNTS_PATH, TRUE_SPEED_PATH, 
                        TIME_PATH, LINK_LENGTH_PATH, TRAFFIC_COUNTS_PATH, 
                        TRANSFORM_PATH, PRED_SPEED_PATH, AADT_PRED_PATH):
    

    model_list = get_model_list(MODELS_PATH)

    df_processed_vehicle_counts_list = load_and_process_vehicle_counts(VEHICLE_COUNTS_PATH)

    df_true_speed_list, if_true_speed = load_true_speed(TRUE_SPEED_PATH)

    print("Using true speed: {}".format(if_true_speed))

    df_speed_estimate = load_speed_estimate(PRED_SPEED_PATH)

    df_link_length_list = load_link_lengths(LINK_LENGTH_PATH)

    traffic_counts(df_processed_vehicle_counts_list, df_true_speed_list, 
                   df_link_length_list, TRAFFIC_COUNTS_PATH, 
                   df_speed_estimate=df_speed_estimate, TRUE_SPEED=if_true_speed, MODE='none')


    df_transform_list = load_transform(TRANSFORM_PATH)

    df_processed_vehicle_counts_list = transform_counts(df_transform_list, df_processed_vehicle_counts_list)

    df_time_list = load_time(TIME_PATH)

    df_aadt_features_list = concatenate_inputs(df_processed_vehicle_counts_list, df_true_speed_list, 
                                               df_time_list, if_true_speed=if_true_speed)

    i = 0

    for df_aadt_features in df_aadt_features_list:

        for model in model_list:

            if 'image_id' in df_aadt_features.columns:

                if df_aadt_features.iloc[0]['image_id'] == model.name:

                    df_aadt_features = df_aadt_features.drop(['image_id'], axis=1)

                    print("Local Authority Count Site: {} \n\nInput Features: \n {}\n".format(model.name, df_aadt_features))

                    x = torch.tensor(df_aadt_features.iloc[0].values, dtype=torch.float32).float()

                    print(x)

                    y = np.round(model(x).detach().numpy(), 2)

                    print("AADT Prediction: {}".format(y))

                    print("Saving to csv: {}".format(AADT_PRED_PATH+'aadt_'+model.name+'.csv'))

                    save_float_to_csv(y, COLUMN_NAMES, model.name, AADT_PRED_PATH+'aadt_'+model.name+'.csv')

                    i = i + 1

                    print("---------------------------------------")
                #else:
                #print("df does not have image_id column!")

                #print("---------------------------------------")

            print("Number of predictions made: {}".format(i))

    return True
