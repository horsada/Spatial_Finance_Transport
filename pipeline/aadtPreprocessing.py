import os
from glob import glob
import pandas as pd

NORMALISE = True
DROP_UNNORMALISE = False
REMOVE_METADATA = False
CHOSEN_COUNT_SITES = [('Luton', 'M1/2557A', 'M1/2557B'), ('Hounslow', 'M4/2188A', 'M4/2188B'), ('Enfield', 'M25/5441A', 'M25/5441B'),
                      ('Blackburn with Darwen', '30361033', '30361032'), ('Havering', 'M25/5790A', 'M25/5790B'), ('Trafford', 'M60/9083A', 'M60/9086B')]


def load_motor_vehicle_data(AADT_PATH):
    pattern = os.path.join(AADT_PATH, 'all_motor_vehicles_*.csv')
    motor_vehicle_file_paths = [os.path.join(AADT_PATH, os.path.basename(x)) for x in glob.glob(pattern)]
    print("LA motor vehicle files: {}".format(motor_vehicle_file_paths))

    df_motor_vehicle_list = []

    for i in range(len(motor_vehicle_file_paths)):
        df_motor_vehicle = pd.read_csv(motor_vehicle_file_paths[i])
        df_motor_vehicle = df_motor_vehicle.loc[:, ~df_motor_vehicle.columns.str.contains('^Unnamed')]

        df_motor_vehicle_list.append(df_motor_vehicle)

    return df_motor_vehicle_list


def merge_dfs_from_lists(df_aadt_list, df_motor_vehicle_list, la_and_count_sites):
    merged_aadt_df_list = []

    for aadt_df in df_aadt_list:
        for motor_vehicle_df in df_motor_vehicle_list:
            for site in la_and_count_sites:
                (la_name, site_a, site_b) = site

                if la_name == motor_vehicle_df.iloc[0]['Local Authority'] and (
                        (site_a == aadt_df.iloc[0]['site_name']) or (site_b == aadt_df.iloc[0]['site_name'])):

                    print("Entered if statement: {} {} {}".format(la_name, site_a, site_b))

                    year = aadt_df.iloc[0]['year']

                    all_motor_vehicles = motor_vehicle_df.loc[motor_vehicle_df['year'] == year].iloc[0][
                        'all_motor_vehicles']
                    cars_and_taxis = motor_vehicle_df.loc[motor_vehicle_df['year'] == year].iloc[0]['cars_and_taxis']
                    buses_and_coaches = motor_vehicle_df.loc[motor_vehicle_df['year'] == year].iloc[0][
                        'buses_and_coaches']
                    lgvs = motor_vehicle_df.loc[motor_vehicle_df['year'] == year].iloc[0]['lgvs']
                    all_hgvs = motor_vehicle_df.loc[motor_vehicle_df['year'] == year].iloc[0]['all_hgvs']

                    print("year: {}, all_motor_vehicles: {}".format(year, all_motor_vehicles))

                    merged_aadt_df = aadt_df.copy()

                    merged_aadt_df.name = 'aadt_' + la_name + '_' + aadt_df.iloc[0]['site_name'].replace('/', '_') + '_' + str(year)

                    merged_aadt_df['cars_and_taxis'] = cars_and_taxis
                    merged_aadt_df['buses_and_coaches'] = buses_and_coaches
                    merged_aadt_df['lgvs'] = lgvs
                    merged_aadt_df['all_hgvs'] = all_hgvs
                    merged_aadt_df['all_motor_vehicles'] = all_motor_vehicles

                    merged_aadt_df['Local Authority'] = motor_vehicle_df['Local Authority']
                    merged_aadt_df['site_name'] = merged_aadt_df['site_name'].astype(str)

                    merged_aadt_df_list.append(merged_aadt_df)

    return merged_aadt_df_list


def normalise_v2(clean_report, PROCESSED_PATH):
    integer_cols = ['0-520cm', '521-660cm', '661-1160cm', '1160+cm', 'total_volume']
    df_transform = pd.DataFrame(index=['min', 'max'], columns=integer_cols)
    site_name = clean_report.iloc[0]['site_name']

    for col_name in integer_cols:
        new_col_name = f"{col_name}_normalised"
        if df_transform is not None:
            min_val = clean_report[col_name].min()
            max_val = clean_report[col_name].max()
            df_transform.loc['min', col_name] = min_val
            df_transform.loc['max', col_name] = max_val
        else:
            min_val = clean_report[col_name].min()
            max_val = clean_report[col_name].max()

        clean_report.loc[:, new_col_name] = (clean_report[col_name] - min_val) / (max_val - min_val)

    print("saving normalization values to: {}".format(PROCESSED_PATH + 'transform_' + site_name.replace('/', '_') + '.csv'))
    df_transform.to_csv(PROCESSED_PATH + 'transform_' + site_name.replace('/', '_') + '.csv')

    return clean_report


def drop_unnormalise(df):
    interger_cols = ['0-520cm', '521-660cm', '661-1160cm', '1160+cm', 'total_volume']
    df = df.drop(columns=interger_cols, axis=1)
    return df


def drop_metadata(df):
    metadata = ['site_id', 'time_period_ending', 'time_interval', 'daily_count', 'report_date', 'timestamp']
    df = df.drop(columns=metadata, axis=1)
    return df


def load_aadt_data(AADT_PATH, PROCESSED_PATH):
    pattern = os.path.join(AADT_PATH, 'aadt_*.csv')
    aadt_file_paths = [os.path.join(AADT_PATH, os.path.basename(x)) for x in glob(pattern)]
    print("AADT files: {}".format(aadt_file_paths))
    df_aadt_list = []

    for i in range(len(aadt_file_paths)):
        df_aadt = pd.read_csv(aadt_file_paths[i])
        df_aadt = df_aadt.dropna()
        df_aadt = df_aadt.loc[:, ~df_aadt.columns.str.contains('^Unnamed')]
        df_aadt['site_name'] = df_aadt['site_name'].astype(str)

        if NORMALISE:
            df_aadt = normalise_v2(df_aadt, PROCESSED_PATH)

        if DROP_UNNORMALISE:
            df_aadt = drop_unnormalise(df_aadt)

        if REMOVE_METADATA:
            df_aadt = drop_metadata(df_aadt)

        df_aadt_list.append(df_aadt)

    return df_aadt_list


def load_motor_vehicle_data(AADT_PATH):
    pattern = os.path.join(AADT_PATH, 'all_motor_vehicles_*.csv')
    motor_vehicle_file_paths = [os.path.join(AADT_PATH, os.path.basename(x)) for x in glob(pattern)]
    print("LA motor vehicle files: {}".format(motor_vehicle_file_paths))
    df_motor_vehicle_list = []

    for i in range(len(motor_vehicle_file_paths)):
        df_motor_vehicle = pd.read_csv(motor_vehicle_file_paths[i])
        df_motor_vehicle = df_motor_vehicle.loc[:, ~df_motor_vehicle.columns.str.contains('^Unnamed')]
        df_motor_vehicle_list.append(df_motor_vehicle)

    return df_motor_vehicle_list


def aadt_preprocess(aadt_path, processed_path):
    df_aadt_list = load_aadt_data(aadt_path, processed_path)
    df_motor_vehicle_list = load_motor_vehicle_data(aadt_path)
    merged_aadt_df_list = merge_dfs_from_lists(df_aadt_list, df_motor_vehicle_list, CHOSEN_COUNT_SITES)

    print("Number of sites with motor vehicles merged: {}".format(len(merged_aadt_df_list)))

    for merged_df in merged_aadt_df_list:
        print("merged df length: {}".format(len(merged_df)))
        merged_df.to_csv(processed_path + merged_df.name + '.csv')