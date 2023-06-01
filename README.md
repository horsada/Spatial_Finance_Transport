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

- roadCharacteristics:
    - Road extraction using DL
    - Automated road segment length and width information extraction using clustering technique
    - **Not currently used in implementation**

- speedEstimation:
    - Average vehicle speed estimation using PCA-based method and the time lag between satellite image bands (MS1 and MS2)
    - To provide additional flexibility to the pipeline for when live speed data isn't available

- AADT:
    - Training and inference of ANN for predicting AADT using information from many parts of the pipeline

- GHGEmissions:
    - GHG Emissions training and inference using information from many parts of the pipeline
    
- models:
    - Saved trained models

- data:
    - Storing of true data, predicted data, and results

- admin:
    - Storing of results from different configurations of the pipeline, for example with and without speed estimation. 
