# Spatial Finance for Sustainable Development (Transport)
## Overview

The full project pipeline is shown in the figure. The input to the pipeline is a satellite image that contains a motorway section with traffic count data. The final output is road transport GHG emissions prediction at the Local Authority level:
- AADT predictions for road segment
- GHG Emissions predictions based on AADT prediction for road segment
- GHG Emissions of local authority area based on road segment GHG emissions prediction

![alt text](https://github.com/horsada/Spatial_Finance_Transport/blob/main/images/FYP_Project_Plan.svg)

Each box in the project pipeline figure indicates a distinct model input/output in the pipeline. The idea is that each box would be isolated
from the rest of the pipeline. This allows for flexiblity in the implementation of each aspect of the pipeline. For example, satellite image preprocessing can
be fully automated, part-automated, or fully manual, depending on certain factors, for example time constraints, performance requirements, etc.

The repo folder structure is as follows:
- Spatial_Finance_Transport: For motorway road transport and GHG emissions, as well as any global implementations (e.g. object detection). This is because the project's initial scope was only for motorway's, but has since expanded to include A roads and minor roads. 
- Spatial_Finance_Transport/ARoads: For code and data specific to A-roads
- Spatial_Finance_Transport/minorRoads: For code and data specific to minor roads

A brief summary of each folder and its contents/purpose is shown below.
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
