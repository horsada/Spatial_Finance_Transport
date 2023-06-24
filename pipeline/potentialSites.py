import urllib.request, json
import pandas as pd
import math
import pickle
import collections
from tqdm import tqdm
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import statsmodels.api as sm

df_aadt_list = []

for id in LOCAL_AUTHORITY_IDS:
    df_aadt_list.append(pd.read_csv('https://storage.googleapis.com/dft-statistics/road-traffic/downloads/aadfbydirection/local_authority_id/dft_aadfbydirection_local_authority_id_{}.csv'.format(id)))

print(df_aadt_list[0].columns)

print(df_aadt_list[0]['count_point_id'].unique())

df_aadt_list[0].head()