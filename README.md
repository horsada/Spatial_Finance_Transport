# Spatial Finance for Sustainable Development (Transport)
## Overview

The full project pipeline is shown in the figure. The input to the pipeline is a satellite image that contains a motorway section with traffic count data. The final output is road transport GHG emissions prediction at the Local Authority level:
- AADT predictions for road segment
- GHG Emissions predictions based on AADT prediction for road segment
- GHG Emissions of local authority area based on road segment GHG emissions prediction

Unfortunately, European Space Imaging rules prohibit the use of extracted satellite data for use other than the approved project. Thus, this readme.md file will explain the pipeline as best as possible given this limitation. In addition, the notebooks are kept with their outputs for interested readers. 

![alt text](https://github.com/horsada/Spatial_Finance_Transport/blob/main/images/FYP Full Pipeline.svg)

Each box in the project pipeline figure indicates a distinct model input/output in the pipeline. The idea is that each box would be isolated
from the rest of the pipeline. This allows for flexiblity in the implementation of each aspect of the pipeline. For example, satellite image preprocessing can
be fully automated, part-automated, or fully manual, depending on certain factors, for example time constraints, performance requirements, etc.

The repo folder structure is as follows:
- Spatial_Finance_Transport: For motorway road transport and GHG emissions, as well as any global implementations (e.g. object detection). This is because the project's initial scope was only for motorway's, but has since expanded to include A roads and minor roads. 
- Spatial_Finance_Transport/ARoads: For code and data specific to A-roads
- Spatial_Finance_Transport/minorRoads: For code and data specific to minor roads

A high-level summary of each directory and its contents/purpose is shown below.
- dataAPI:
    - Notebooks to retrieve information on traffic and GHG emissions in the UK.
    - Function to extract information (e.g. specific sites from langtitude and longtitude) and feature extraction
    - Primary data sources used for training ANN's

- objectDetection:
    - YOLOv5 vehicle object detection training and validation
    - Also includes training notebook for a SWIN backbone RetinaNet model, which is not used in the implementation

- roadCharacteristics:
    - Road extraction using DL
    - Automated road segment length and width information extraction using clustering technique
    - **Not currently used in implementation**

- AADT:
    - Pre-processing data for ANN compatability
    - Training and inference of ANN for predicting AADT using information from many parts of the pipeline
    
- models:
    - Saved trained models (AADT and object detection)

- data:
    - Storing of true data, predicted data, results and plots

- admin:
    - Storing of results from different configurations of the pipeline, for example with and without speed estimation. 

- implementation:
   - Generation of results for object detection, 15-minute traffic counts, speed estimation, AADT and emissions

## Data

The data folder is described in more detail in this section and a Luton Local Authority examples is given to help understanding.

data/ground_truth_data: for storing ground truth data

data/ground_truth_data/speed_data: for storing average speeds. Example:
```
image_id, avg_mph
luton_m1_2557a, 65
```

data/ground_truth_data/time_data: for storing time data. Example:
```
image_id, day, month, hour
luton_m1_2557a, 24, 2, 11
```

data/ground_truth_data/link_length_data: for storing link lengths. Example:
```
image_id, link_length
luton_m1_2557a, 1.35
```

data/ground_truth_data/aadt: for storing true aadt (historical year). Example:
```
site_name, site_id, report_date, time_period_ending, time_interval, 0-520cm, 521-660cm, ..., aadt
M1/2557A, 332, 2017-01-01, 0:14:00, 0, 67, 24, 70000
```

data/satellite_images: for storing raw satellite images. Example:
```
luton_m1_2557a.tif
```

data/predicted: for storing predicted data.


data/predicted/aadt: predicted aadt (test year). Example:
```
image_id, aadt, cars_and_taxis, buses_and_coaches, lgvs, all_hgvs
luton_m1_2557a, 70000, 59850, 0, 12636, 11988
```

data/predicted/ghg_emissions: predicted ghg emissions (test year). Example:
```
image_id, ghg_emissions
luton_m1_2557a, 25
```

data/predicted/speed: predicted average speeds. Example:
```
image_id, speed_estimates
luton_m1_2557a, 0
```

data/predicted/traffic_counts: predicted traffic counts. Example:
```
image_id, Total_N15, Small_N15, Medium_N15, Large_N15, Very Large_N15
luton_m1_2557a, 565, 120, 108, 204, 132
```

data/predicted/vehicle_counts: predicted vehicle_counts. Example:
```
image_id, x_min, x_max, y_min, y_max, category_name, area
luton_m1_2557a, 583, 599, 1152, 1167, Small Car, 244
```


## Implementation

The following files relate directly to the satellite image and LA's tested and so are considered implementation.

- Potential Sites
    - mostly Exploratory Data Analysis, but also preprocesses LA AADT and emissions data 
- AADT implementation (Multi for when vehicle type data is available)
- GHG Emissions implementation (Multi for when vehicle type data is available)
- speedEstimation
    - Average vehicle speed estimation using PCA-based method and the time lag between satellite image bands (MS1 and MS2)
    - To provide additional flexibility to the pipeline for when live speed data isn't available
- evaluation
    - Plotting graphs, calculating metrics, and doing comparisons between pipeline configurations 

## Using the Pipeline

The repo can be used in the following order to generate results, assuming the satellite image(s) have already been pre-processed.

1. dataAPI/trafficAPI: Download count sites (defined by latitude and longtitude) from UK highways england API
2. dataAPI/GHGEmissionsAPI: Download LA emissions data by road type (Motorways, A-Roads, Minor Roads)
3. implementation/potentialSites: To generate AADT and GHG Emissions data. Here, we can choose which AADT statistic to use to represent the LA distribution, e.g. max, median mean... for each vehicle type. This can also be used to do EDA on the downloaded data from 1, 2. 
4. AADT/AADTPreProcessing: To pre-process the downloaded AADT data for ANN compatability
5. AADT/MultiAADTTraining: Training and validation of ANN models. Can specify model architecture and hyperparameters here. The ANN target variable are the AADT statistic chosen in 3. We also save transformation values for use in AADT prediction
6. implementation/multiAADTImplementation: To do AADT prediction. Here we load speed data, time data, saved model weights and traffic data transformations to perform predictions.
7. implementation/multiGHGEmissions: To do GHG emissions prediction. Here we define specific fuel consumption, fuel distributions (petrol, diesel etc.) and motorway lengths in each LA. 
8. implementation/evaluation: To evaluate each output of the pipeline. Most require ground truth data 


## Future Work

Work is ongoing to allow the pipeline to be run from a single command line input:
```
python main.py --la LA --cs_id CS_ID
```
Where the user would add to the following directories as required data:
- data/satellite_images: raw satellite images in .tif format
- data/ground_truth_data/speed_data: average speeds from satellite images (if available) in .csv format
- data/ground_truth_data/time_data: month, day, hour from satellite images in .csv format
- data/ground_truth_data/link_length_data: Length of road in each satellite image in .csv format
CSV files must contain a image_id column that matches the name of the satellite image file names. These user inputted CSV files only have 1 row.

The LA argument is specifying the Local Authority and the CS_ID is specifying the count site to use within the LA. 
