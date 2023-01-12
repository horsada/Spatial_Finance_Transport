# Spatial Finance for Sustainable Development (Transport)
## Overview

The project pipeline is shown in the figure. The input to the pipeline is a satellite image. The most important outputs are the following:
- AADT predictions for road segment
- GHG Emissions predictions based on AADT prediction for road segment
- GHG Emissions of local authority area based on road segment GHG emissions prediction

![alt text](https://github.com/horsada/Spatial_Finance_Transport/blob/main/images/Project_Plan.PNG)

Each box in the project pipeline figure indicates a distinct model input/output in the pipeline. The idea is that each box would be isolated
from the rest of the pipeline. This allows for flexiblity in the implementation of each aspect of the pipeline. For example, satellite image preprocessing can
be fully automated, part-automated, or fully manual, depending on certain factors, for example time constraints, performance requirements, etc.

To get an idea of the full pipeline, <kbd> main.ipynb </kbd> shows the most automated and generalised implementation of the pipeline.

A brief summary of each folder and its contents/purpose is shown below.
- dataAPI:
    - Notebooks to retrieve information on traffic and GHG emissions in the UK.
    - Function to extract information (e.g. specific sites from langtitude and longtitude) and feature extraction
    - Primary data sources used for training ANN's

- objectDetection:
    - YOLOv5 vehicle object detection
    - Includes training and inference

- roadCharacteristics:
    - Road extraction using DL
    - Automated road segment length and width information extraction using clustering technique
    - Used to improve AADT and GHG Emissions prediction

- speedEstimation:
    - Average vehicle speed estimation using time lag between satellite image bands (MS1 and MS2)
    - Used to improve AADT prediction

- AADT:
    - Training and inference of ANN for predicting AADT using information from many parts of the pipeline

- GHGEmissions:
    - GHG Emissions training and inference using information from many parts of the pipeline
    
- models:
    - Saved trained model and state dict's
