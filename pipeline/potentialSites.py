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
import plotly.io as pio

import statsmodels.api as sm

LOCAL_AUTHORITY_IDS = ['120', '69', '111', '82', '137', '166', '121', '156', '57', '189', '152', '91', '180',
                       '201', '161', '198', '94', '160']

LOCAL_AUTHORITY_NAMES = ['Luton', 'Worcester', 'Hounslow', 'Portsmouth', 'Southampton', 'South Tyneside', 'Enfield', 'Halton', 'Barnet', 'Dudley', 'Coventry',
                         'Trafford', 'Bracknell Forest', 'Havering', 'Sunderland', 'Liverpool', 'Blackburn with Darwen',
                         'Bolton']

UNWANTED_COLS = ['region_name', 'region_id', 'road_type', 'pedal_cycles', 'two_wheeled_motor_vehicles', 'hgvs_2_rigid_axle',
                    'hgvs_3_rigid_axle', 'hgvs_4_or_more_rigid_axle',
                    'hgvs_3_or_4_articulated_axle', 'hgvs_5_articulated_axle',
                    'hgvs_6_articulated_axle', 'start_junction_road_name', 'end_junction_road_name', 'easting',
                    'northing', 'estimation_method', 'estimation_method_detailed', 'link_length_miles']

def clean_aadt_list(df_aadt_list):

    for i in range(len(df_aadt_list)):
        df = df_aadt_list[i]
        df_motorways = df[df['road_name'].str.contains('M')]
        df_2005_onwards = df_motorways[df_motorways['year'] >= 2005]
        df_2005_onwards =  df_2005_onwards[df_2005_onwards['year'] != 2021]
        df_aadt_list[i] = df_2005_onwards.drop(UNWANTED_COLS, axis=1, errors='ignore')

    return df_aadt_list

def group_aadt_list(df_aadt_list):

    grouped_df_aadt_list = []

    for i in range(len(df_aadt_list)):
        df = df_aadt_list[i]
        #grouped_df = df.groupby(by=['count_point_id', 'year']).mean()
        local_authority_name = df.iloc[0]['local_authority_name']
        grouped_df = df.groupby(by=['year']).quantile(1.00)
        grouped_df['Calendar Year'] = grouped_df.index
        grouped_df = grouped_df.reset_index()
        grouped_df['Local Authority'] = local_authority_name
        grouped_df_aadt_list.append(grouped_df)

    return grouped_df_aadt_list

def merge_aadt_ghg_list(df_aadt_list, df_ghg_list):

    merged_dfs_list = []

    for i, aadt_df in enumerate(df_aadt_list):
        for j, ghg_df in enumerate(df_ghg_list):
            if aadt_df.iloc[0]['Local Authority'] == ghg_df.iloc[0]['Local Authority']:
                merged_df = aadt_df.merge(ghg_df, on=['Calendar Year', 'Local Authority'])
                merged_dfs_list.append(merged_df)

    return merged_dfs_list


def plot_aadt_ghg(merged_dfs_list, show_plot=True, show_ols=True):
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly * 2

    df_ols = merged_dfs_list[0][['Calendar Year']].copy()

    df_total = merged_dfs_list[0][['all_motor_vehicles', 'Annual Territorial emissions (kt CO2e)']].copy()

    for i, df in enumerate(merged_dfs_list):
        la_name = df.iloc[0]['Local Authority']
        fig = fig.add_trace(go.Scatter(x = df['all_motor_vehicles'], y=df['Annual Territorial emissions (kt CO2e)'], mode='markers', name=la_name, marker=dict(color=colors[i])))

        x = sm.add_constant(df['all_motor_vehicles'])
        model = sm.OLS(df['Annual Territorial emissions (kt CO2e)'], x).fit()
        df_ols[la_name+ '_OLS'] = model.fittedvalues

        color = fig.data[0].line.color
        # regression data
        if show_ols:
            fig.add_trace(go.Scatter(x=df['all_motor_vehicles'],
                                y=df_ols[la_name + '_OLS'],
                                mode='lines',
                                name=la_name+' OLS',
                                marker=dict(color=colors[i],size=15)
                                ))

        if i > 0:
            df_total = pd.concat([df_total, df[['all_motor_vehicles', 'Annual Territorial emissions (kt CO2e)']]], ignore_index=True)


    df_total = df_total.dropna()

    x = sm.add_constant(df_total['all_motor_vehicles'])
    model = sm.OLS(df_total['Annual Territorial emissions (kt CO2e)'], x).fit()
    df_total['Total_OLS'] = model.fittedvalues

    fig.add_trace(go.Scatter(x=df_total['all_motor_vehicles'],
                            y=df_total['Total_OLS'],
                            mode='lines',
                            name='Total OLS',
                            line=dict(color='black',width=3)
                            ))

    if show_plot:
        fig.update_xaxes(title='AADT', title_font=dict(size=30), tickfont=dict(size=24))
        fig.update_yaxes(title='Annual Territorial emissions (kt CO2e)', title_font=dict(size=30), tickfont=dict(size=24))

        fig.update_layout(
            title='Motorway AADT vs GHG Emissions per year per LA',
            autosize=False,
            width=1000,
            height=1000,
            legend=dict(font=dict(size=24)))

        # Save the figure as SVG
        pio.write_image(fig, 'la_aadt_ghg_scatter_plot.svg', format='svg')
        fig.show()

    return df_ols, df_total

def potential_sites(CHOSEN_COUNT_SITES, AADT_DATA_PATH, GHG_PROCESSED_PATH, GHG_EMISSIONS_DATA_PATH):

    df_aadt_list = []

    for id in LOCAL_AUTHORITY_IDS:
        df_aadt_list.append(pd.read_csv('https://storage.googleapis.com/dft-statistics/road-traffic/downloads/aadfbydirection/local_authority_id/dft_aadfbydirection_local_authority_id_{}.csv'.format(id)))

    print(df_aadt_list[0].columns)

    print(df_aadt_list[0]['count_point_id'].unique())

    df_aadt_list = clean_aadt_list(df_aadt_list)

    grouped_df_aadt_list = group_aadt_list(df_aadt_list)

    df_ghg = pd.read_csv(GHG_PROCESSED_PATH, index_col=0)

    df_ghg = df_ghg[df_ghg['Local Authority'].isin(LOCAL_AUTHORITY_NAMES)]

    df_ghg_list = [d for _, d in df_ghg.groupby(['Local Authority'])]

    merged_dfs_list = merge_aadt_ghg_list(grouped_df_aadt_list, df_ghg_list)

    chosen_site_names = [item[0] for item in CHOSEN_COUNT_SITES]

    print("Chosen site names: {}".format(chosen_site_names))

    chosen_merged_dfs_list = []

    for df in merged_dfs_list:
        if df.iloc[0]['Local Authority'] in chosen_site_names:
            chosen_merged_dfs_list.append(df)

    print("number of chosen LA's: {}".format(len(chosen_merged_dfs_list)))

    for df in chosen_merged_dfs_list:
        df_la_aadt = df[['year', 'cars_and_taxis', 'buses_and_coaches',	'lgvs', 'all_hgvs', 'all_motor_vehicles', 'Local Authority']]
        df_la_ghg = df[['year', 'Annual Territorial emissions (kt CO2e)', 'Local Authority']]

        df_la_aadt.name = 'all_motor_vehicles_'+df_la_aadt.iloc[0]['Local Authority']
        df_la_ghg.name = 'ghg_emissions_'+df_la_ghg.iloc[0]['Local Authority']

        df_la_aadt.to_csv(AADT_DATA_PATH+df_la_aadt.name+'.csv')
        df_la_ghg.to_csv(GHG_EMISSIONS_DATA_PATH+df_la_ghg.name+'.csv')