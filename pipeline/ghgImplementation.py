import os
import pandas as pd
import numpy as np

VEHICLE_CATEGORIES = ['Passenger Vehicle',
  'Small Car',
  'Bus',
  'Pickup Truck',
  'Utility Truck',
  'Truck',
  'Cargo Truck',
  'Truck w/Box',
  'Truck Tractor',
  'Trailer',
  'Truck w/Flatbed',
  'Truck w/Liquid',
  'Passenger Car'
]

EMISSIONS_CATEGORY_MAPPING = {
    'Passenger Vehicle': 'Petrol cars',
    'Small Car': 'Petrol cars',
    'Pickup Truck': 'Petrol LGVs',
    'Utility Truck': 'Petrol LGVs',
    'Truck': 'Petrol LGVs',
    'Cargo Truck': 'Rigid HGVs',
    'Truck Tractor': 'Rigid HGVs',
    'Trailer': 'Petrol LGVs',
    'Truck w/Flatbed': 'Rigid HGVs',
    'Truck w/Liquid': 'Rigid HGVs',
    'Passenger Car': 'Petrol cars',
    'Truck w/Box': 'Petrol LGVs',
    'Bus': 'Buses',
    'Trailer': 'Petrol LGVs',
    'Cargo Car': 'Petrol LGVs'
}


# https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1040512/env0103.ods
# https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/749248/env0104.ods
VEHICLE_KM_PER_LITRE_MAPPING = {
    'aadt': 15,
    'cars_and_taxis': 20,
    'buses_and_coaches': 3,
    'lgvs': 15,
    'all_hgvs': 3.6
}

VEHICLE_EMISSIONS_FACTORS_MAPPING = {
    'aadt': 'Petrol cars',
    'cars_and_taxis': 'Petrol cars',
    'buses_and_coaches': 'Buses',
    'lgvs': 'Petrol LGVs',
    'all_hgvs': 'Rigid HGVs'
}

KG_TO_KT = 1e-6

# https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2022
# Total kg CO2e per unit litre
PETROL = 2.16
DIESEL = 2.56

# https://www.gov.uk/government/statistical-data-sets/vehicle-licensing-statistics-data-tables
PETROL_DIESEL_AVERAGE = 0.36 * DIESEL + 0.64 * PETROL


# km
LUTON_ROAD_LENGTH = 4.18
BLACKBURN_ROAD_LENGTH = 12.87
HOUNSLOW_ROAD_LENGTH = 15.77
HAVERING_ROAD_LENGTH = 19
TRAFFORD_ROAD_LENGTH = 9.98


def save_float_to_csv(float_value, column_name, image_id, file_name):
    """
    Save a float value to a CSV file with the specified column name and file name.

    Args:
        float_value (float): The float value to be saved.
        column_name (str): The name of the column in the CSV file.
        file_name (str): The name of the CSV file to be saved.
    """
    # Create a DataFrame with a single row and the specified column name and value
    df = pd.DataFrame({'image_id': image_id, column_name: [float_value]})

    # Save the DataFrame to a CSV file
    df.to_csv(file_name, index=False)


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


def convert_category_names(dataframes_list, mapping_dict):
    """
    Convert category names in a list of DataFrames using a mapping dictionary.

    Args:
        dataframes_list (list): List of DataFrames with 'category_name' column.
        mapping_dict (dict): Dictionary containing mapping of old category names to new category names.

    Returns:
        list: List of DataFrames with updated category names.
    """
    updated_dataframes = []
    for df in dataframes_list:
        df['Vehicle Type'] = df['category_name'].map(mapping_dict)
        updated_dataframes.append(df)
    return updated_dataframes



def calculate_ghg_emissions(car_category, emission_factors, road_length):
    """
    Calculates GHG emissions for a given car category, emission factors, and road length.

    Args:
        car_category (str): Category of the car (e.g., "Small Car", "Midsize Car", etc.).
        emission_factors (dict): Dictionary containing emission factors for different car categories.
        road_length (float): Length of the road segment in kilometers.

    Returns:
        float: Total GHG emissions in kilograms for the given car category and road length.
    """
    # Check if the emission factors dictionary contains the given car category
    if car_category not in emission_factors:
        raise ValueError("Car category not found in emission factors dictionary.")

    # Get the emission factors for the given car category
    car_emission_factors = emission_factors[car_category]

    # Calculate GHG emissions using the emission factors and road length
    ghg_emissions = car_emission_factors['co2'] * road_length + \
                    car_emission_factors['ch4'] * road_length + \
                    car_emission_factors['n2o'] * road_length

    return ghg_emissions



def add_total_column(dataframes_list, other_dataframe):
    """
    Add a 'Total' column from one DataFrame to each DataFrame in a list of DataFrames based on the 'Vehicle Type' column.

    Args:
        dataframes_list (list): List of DataFrames.
        other_dataframe (DataFrame): DataFrame to extract the 'Total' column from.

    Returns:
        list: List of DataFrames with the 'Total' column added.
    """
    updated_dataframes = []
    for df in dataframes_list:
        if 'Vehicle Type' in df.columns and 'Vehicle Type' in other_dataframe.columns:
            total_column = other_dataframe[['Vehicle Type', 'Total']]
            df = df.merge(total_column, on='Vehicle Type', how='left')
        updated_dataframes.append(df)
    return updated_dataframes


def create_vehicle_type_counts_df(df):
    """
    Count unique vehicle types in a DataFrame and return a DataFrame with columns as vehicle types and
    a single row with counts as values.

    Args:
        df (pandas.DataFrame): DataFrame containing the columns: image_id, x_min, x_max, y_min, y_max,
                               category_name, area, Vehicle Type, and Total.

    Returns:
        pandas.DataFrame: DataFrame with columns as vehicle types and a single row with counts as values.
    """
    # Check if "Vehicle Type" column is present in the DataFrame
    if "Vehicle Type" not in df.columns:
        raise ValueError("Column 'Vehicle Type' not found in the DataFrame.")

    # Count unique values in "Vehicle Type" column
    vehicle_type_counts = df["Vehicle Type"].value_counts().to_dict()

    # Create a DataFrame from the counts dictionary
    counts_df = pd.DataFrame(vehicle_type_counts, index=[0])

    return counts_df

def load_pred_vehicle_counts(VEHICLE_COUNTS_PATH):

    dfs = []

    vehicle_count_paths = get_files_in_directory(VEHICLE_COUNTS_PATH)

    for vehicle_count_path in vehicle_count_paths:
        df = pd.read_csv(vehicle_count_path)

        df.name = df.iloc[0]['image_id']
        print(df.name)
        dfs.append(df)

    return dfs

def load_pred_aadt(AADT_PATH, dfs):

    aadt_paths = get_files_in_directory(AADT_PATH)

    for aadt_path in aadt_paths:
        df_aadt = pd.read_csv(aadt_path, sep = ',', skipinitialspace = True)

        for df in dfs:
            if df.iloc[0]['image_id'] == df_aadt.iloc[0]['image_id']:
                
                print("found match for: {}".format(df.iloc[0]['image_id']))
                df['aadt'] = df_aadt.iloc[0]['aadt']
                df['cars_and_taxis'] = df_aadt.iloc[0]['cars_and_taxis']
                df['buses_and_coaches'] = df_aadt.iloc[0]['buses_and_coaches']
                df['lgvs'] = df_aadt.iloc[0]['lgvs']
                df['all_hgvs'] = df_aadt.iloc[0]['all_hgvs']

    return dfs

def ghg_implementation(AADT_PRED_PATH, VEHICLE_COUNTS_PATH, GHG_EMISSIONS_PRED_PATH):

    dfs = load_pred_vehicle_counts(VEHICLE_COUNTS_PATH)

    dfs = load_pred_aadt(AADT_PRED_PATH, dfs)

    total_emissions = []

    for df in dfs:
        la_name_id = df.iloc[0]['image_id']
        ghg_emissions = 0
        LENGTH = 0

        print(la_name_id)

        if 'aadt' in df:

            aadt = df.iloc[0]['aadt']
            cars_and_taxis = df.iloc[0]['cars_and_taxis']
            buses_and_coaches = df.iloc[0]['buses_and_coaches']
            lgvs = df.iloc[0]['lgvs']
            all_hgvs = df.iloc[0]['all_hgvs']

            if la_name_id.find('blackburn') != -1:
                LENGTH = BLACKBURN_ROAD_LENGTH

            elif la_name_id.find('luton') != -1:
                LENGTH = LUTON_ROAD_LENGTH

            elif la_name_id.find('hounslow') != -1:
                LENGTH = HOUNSLOW_ROAD_LENGTH

            elif la_name_id.find('havering') != -1:
                LENGTH = HAVERING_ROAD_LENGTH

            elif la_name_id.find('trafford') != -1:
                LENGTH = TRAFFORD_ROAD_LENGTH


            # VEHICLE KM TRAVELLED (km)
            aadt_vehicle_km_travel = LENGTH * df.iloc[0]['aadt'] * 365
            cars_and_taxis_vehicle_km_travel = LENGTH * df.iloc[0]['cars_and_taxis'] * 365
            buses_and_coaches_vehicle_km_travel = LENGTH * df.iloc[0]['buses_and_coaches'] * 365
            lgvs_vehicle_km_travel = LENGTH * df.iloc[0]['lgvs'] * 365
            all_hgvs_vehicle_km_travel = LENGTH * df.iloc[0]['all_hgvs'] * 365



            # SPECIFIC FUEL CONSUMPTION (km/litre)
            aadt_vehicle_km_litre = VEHICLE_KM_PER_LITRE_MAPPING['aadt']
            cars_and_taxis_vehicle_km_litre = VEHICLE_KM_PER_LITRE_MAPPING['cars_and_taxis']
            buses_and_coaches_vehicle_km_litre = VEHICLE_KM_PER_LITRE_MAPPING['buses_and_coaches']
            lgvs_vehicle_km_litre = VEHICLE_KM_PER_LITRE_MAPPING['lgvs']
            all_hgvs_vehicle_km_litre = VEHICLE_KM_PER_LITRE_MAPPING['all_hgvs']


            # LITRES USED (litres)
            aadt_litres = aadt_vehicle_km_travel / aadt_vehicle_km_litre # litres
            cars_and_taxis_litres = cars_and_taxis_vehicle_km_travel / cars_and_taxis_vehicle_km_litre # litres
            buses_and_coaches_litres = buses_and_coaches_vehicle_km_travel / buses_and_coaches_vehicle_km_litre # litres
            lgvs_litres = lgvs_vehicle_km_travel / lgvs_vehicle_km_litre # litres
            all_hgvs_litres = all_hgvs_vehicle_km_travel / all_hgvs_vehicle_km_litre # litres


            # EMISSIONS FACTORS (kg CO2e)
            #aadt_emissions_factor = df_emissions_factors.loc[df_emissions_factors['Vehicle Type'] == VEHICLE_EMISSIONS_FACTORS_MAPPING['aadt'], 'Total'].values[0] # kg co2
            #cars_and_taxis_emissions_factor = df_emissions_factors.loc[df_emissions_factors['Vehicle Type'] == VEHICLE_EMISSIONS_FACTORS_MAPPING['cars_and_taxis'], 'Total'].values[0] # kg co2
            #buses_and_coaches_emissions_factor = df_emissions_factors.loc[df_emissions_factors['Vehicle Type'] == VEHICLE_EMISSIONS_FACTORS_MAPPING['buses_and_coaches'], 'Total'].values[0] # kg co2
            #lgvs_emissions_factor = df_emissions_factors.loc[df_emissions_factors['Vehicle Type'] == VEHICLE_EMISSIONS_FACTORS_MAPPING['lgvs'], 'Total'].values[0] # kg co2
            #all_hgvs_emissions_factor = df_emissions_factors.loc[df_emissions_factors['Vehicle Type'] == VEHICLE_EMISSIONS_FACTORS_MAPPING['all_hgvs'], 'Total'].values[0] # kg co2


            # GHG EMISSIONS (kg CO2e)
            aadt_emissions = PETROL_DIESEL_AVERAGE * aadt_litres
            cars_and_taxis_emissions = PETROL_DIESEL_AVERAGE * cars_and_taxis_litres
            buses_and_coaches_emissions = PETROL_DIESEL_AVERAGE * buses_and_coaches_litres
            lgvs_emissions = PETROL_DIESEL_AVERAGE * lgvs_litres
            all_hgvs_emissions = PETROL_DIESEL_AVERAGE * all_hgvs_litres

            # TOTAL EMISSIONS (kg CO2e -> kt CO2e)
            ghg_emissions = np.round(cars_and_taxis_emissions + buses_and_coaches_emissions + lgvs_emissions + all_hgvs_emissions, 1)

            ghg_emissions = ghg_emissions * KG_TO_KT

            print("LA Count Site: {}, AADT Prediction: {}, GHG Emissions Prediction: {}".format(la_name_id, aadt, ghg_emissions))

            save_float_to_csv(ghg_emissions, 'ghg_emissions', image_id=la_name_id, file_name=GHG_EMISSIONS_PRED_PATH+'ghg_emissions_'+la_name_id+'.csv')
            total_emissions.append((la_name_id, ghg_emissions))

            return True