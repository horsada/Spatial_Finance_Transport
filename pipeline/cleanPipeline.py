import os
import fnmatch

def clean_pipeline(root_path):

    exempt_files = [
    os.path.join(root_path, 'data/ground_truth_data/ghg_emissions/GHG_potential_sites.csv')
]

    dirs = [
        root_path + '/data/ground_truth_data/aadt/',
        root_path + '/data/ground_truth_data/aadt/processed/',
        root_path + '/data/ground_truth_data/ghg_emissions/',
        root_path + '/data/ground_truth_data/link_length_data/',
        root_path + '/data/speed_data/',
        root_path + '/data/time_data/',
        root_path + '/data/traffic_counts/',
        root_path + '/data/satellite_images/',
        root_path + '/data/satellite_images/processed/',
        root_path + '/data/predicted/'
    ]

    for directory in dirs:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path not in exempt_files:
                    os.remove(file_path)
                else:
                    print(file_path)


    return True
