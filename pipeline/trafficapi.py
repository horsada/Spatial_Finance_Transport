import urllib.request, json
import pandas as pd
import math
import pickle
import collections
from tqdm import tqdm
import datetime
import os
import numpy as np

"""## Global Variables"""

BLACKBURN_1 = (53.71646, 53.71644, -2.459, -2.4594)
BLACKBURN_2 = (53.7163, 53.71628, -2.4592, -2.4593)

HAVERING_1 = (51.5532, 51.5531, 0.287574, 0.287572)
HAVERING_2 = (51.55328, 51.55326, 0.287381, 0.28737)

LUTON_1 = (51.9036, 51.90358, -0.47733, -0.47735)
LUTON_2 = (51.9036, 51.90358, -0.47698, -0.4770)

HOUNSLOW_1 = (51.4897, 51.4896, -0.3696, -0.3697)
HOUNSLOW_2 = (51.4899, 51.4898, -0.3698, -0.36983)

TRAFFORD_1 = (53.41847, 53.41845, -2.28659, -2.28661)
TRAFFORD_2 = (53.41656, 53.31654, -2.2839, -2.28395)

COUNT_SITE_LOCATIONS = [BLACKBURN_1, BLACKBURN_2, HAVERING_1, HAVERING_2, LUTON_1, LUTON_2, HOUNSLOW_1, HOUNSLOW_2,
                        TRAFFORD_1, TRAFFORD_2]

DATES = [('01012017', '31122017'), ('01012018', '31122018')]

QUALITY_THRESHOLD = 0
NORMALISE_DF = False
NUMBER_OF_DAYS = 30*6

"""## Helper Functions"""

def dt64_to_float(dt64):
    """
    year = dt64.astype('M8[Y]')
    print("year: {}".format(year))
    days = (dt64 - year).astype('timedelta64[D]')
    print("days: {}".format(days))
    year_next = year + np.timedelta64(1, 'Y')
    days_of_year = (year_next.astype('M8[D]') - year.astype('M8[D]')).astype('timedelta64[D]')
    print("days_of_year: {}".format(days_of_year))
    dt_float = 1970 + year.astype(float) + days / (days_of_year)
    print("dt_float: {}".format(dt_float))
    """
    dt_float = dt64.values.astype('float64')
    dt_float = ((dt_float)-dt_float.min())/(dt_float.max()-dt_float.min())
    return dt_float

def df_aadt_final(site_dfs, START_DATE, AADT_PATH, no_days=7, save=True):

    aadt_final_dfs = []

    for i in range(len(site_dfs)):
        site_df = site_dfs[i] # take specific site

        site_df['report_date'] = pd.to_datetime(site_df['report_date'], infer_datetime_format=True)
        site_df['year'] = site_df['report_date'].dt.year
        site_df['month'] = site_df['report_date'].dt.month
        site_df['day'] = site_df['report_date'].dt.day
        site_df['hour'] = site_df['timestamp'].dt.hour

        # Groupby site_id and report_date
        dfs = [site_day for _, site_day in site_df.groupby(['report_date'])] # groupby days

        print("length of dfs (should be 365 for aadt calculations: {}".format(len(dfs)))

        yearly_count = 0

        for i in range(len(dfs)):
            df_site_day = dfs[i] # Take single day
            df_site_day['daily_count'] = df_site_day['total_volume'].sum() # Daily count
            yearly_count += df_site_day.iloc[0]['daily_count']

            # timestamp preprocessing (currently unused)
            #df_site_day['timestamp_min_max'] = dt64_to_float(df_site_day['timestamp'])

        aadt = yearly_count / len(dfs) # in case some days have been removed (otherwise 365)

        for i in range(len(dfs)):
            df_site_day = dfs[i] # Take single day
            df_site_day['aadt'] = aadt # add aadt column

        # Take random 7 days
        random_days = [np.random.randint(low=1, high=len(dfs)) for _ in range(no_days)]
        df_aadt_final = dfs[0].append([dfs[i] for i in random_days], ignore_index=True)

        print("year: {}".format(START_DATE[-4:]))
        print("site id: {}".format(df_aadt_final.iloc[0]['site_id']))

        site_name = df_aadt_final.iloc[0]['site_name']

        print("site name: {}".format(site_name))
        site_name = site_name.replace('/', '_')

        if save:
            print("saving {} to: {}. length of df: {}".format(site_name, AADT_PATH, len(df_aadt_final)))
            df_aadt_final.to_csv(AADT_PATH + 'aadt_{}_year_{}.csv'.format(site_name, START_DATE[-4:]))
        aadt_final_dfs.append(df_aadt_final)

    return aadt_final_dfs

"""## Initial Site Retrieval"""

# Download information about all UK sites
url_text = "https://webtris.highwaysengland.co.uk/api/v1/sites"
with urllib.request.urlopen(url_text) as url:
    data = json.loads(url.read().decode())
sites = data['sites']


"""## Helper Functions"""

def get_quality_area(sites,
                     site_name,
                     max_lat=math.inf,
                     max_long=math.inf,
                     min_lat=-math.inf,
                     min_long=-math.inf,
                     start_date='01062021',
                     end_date = '15062022',
                     quality_threshold = 50):
    '''
    Returns a dataframe of Traffic Count sites in the specified area and time with sufficent reporting quality
            Parameters:
                    max_lat, max_long, min_lat, min_long (int): Coordinates defining the rectangular area of interest. Default is entire globe.
                    start_date, end_date (str): Strings of the start and end dates of our search ddmmyy
                    quality_threshold (int): Only indclude sites that have at least quality_threshold % of times reporting data
            Returns:
                    quality_area_sites_df (dataframe): Report high quality sites, cols are: Id, Name, Description, Longitude, Latitude, Status
    '''
    # Convert sites query into df and filter onto our area
    print("SITE NAME: {}".format(site_name))
    sites_df = pd.DataFrame(data = sites)

    #area_sites_df = sites_df.loc[(min_long < sites_df.Longitude) & (sites_df.Longitude < max_long)
                                #& (min_lat < sites_df.Latitude) & (sites_df.Latitude < max_lat)]

    area_sites_df = sites_df.loc[sites_df['Description'] == site_name]                            
    area_sites_df = area_sites_df.reset_index(drop=True)
    area_ids = list(area_sites_df.Id)
    print("AREA IDS: {}".format(area_ids))

    # Next filter onto sites with good quality data:
    quality_responces = []
    for site_id in tqdm(area_ids):
        url_text = f"https://webtris.highwaysengland.co.uk/api/v1/quality/overall?sites={site_id}&start_date={start_date}&end_date={end_date}"
        with urllib.request.urlopen(url_text) as url:
            responce = json.loads(url.read().decode())
        quality_responces.append(responce)

    # We only want sites with quality greater than threshold
    good_quality_ids = []
    for responce in quality_responces:
        if responce['data_quality'] >= quality_threshold:
            good_quality_ids.append(responce['sites'])

    quality_area_sites_df = area_sites_df.loc[area_sites_df.Id.isin(good_quality_ids)]
    quality_area_sites_df = quality_area_sites_df.reset_index(drop=True)

    print('quality_area_sites_df: {}'.format(quality_area_sites_df))
    return quality_area_sites_df

def daily_report_query_url(site_id, page_num, start_date = '15062021', end_date = '15062022'):
    '''Generates the query url for page page_num of traffic reporting of site site_id'''
    query_url = f"https://webtris.highwaysengland.co.uk/api/v1/reports/Daily?sites={site_id}&start_date={start_date}&end_date={end_date}&page={page_num}&page_size=10000"
    return query_url

def get_site_report(site_id, start_date='15062021', end_date='15062022'):
    '''
    Returns a dataframe of traffic counts on a specified site and date range.
            Parameters:
                    site_id (str): Site's unique id
                    start_date, end_date (str): Strings of the start and end dates of our search ddmmyy
            Returns:
                    report_df (dataframe): Report of traffic counts for that site
                    header (dict): Columns of the dataframe
    '''
    # Download page 1
    report_url = daily_report_query_url(site_id, 1, start_date, end_date)
    with urllib.request.urlopen(report_url) as url:
        report_page = json.loads(url.read().decode())

    # Work out how many pages are required
    header = report_page['Header']
    rows = report_page['Rows']
    row_count = header['row_count']
    total_pages = math.ceil(row_count / 10000)
    # Make a dataframe of the rows so dar
    report_df = pd.DataFrame(data = rows)

    for i in range(2, total_pages+1):
        # Get page i of the report
        report_url = daily_report_query_url(site_id, i, start_date, end_date)
        with urllib.request.urlopen(report_url) as url:
            report_page = json.loads(url.read().decode())

        rows = report_page['Rows']
        current_page_df = pd.DataFrame(data = rows)
        report_df = pd.concat([report_df, current_page_df], ignore_index=True)

    return report_df, header

def get_reports_from_sites_df(sites_df, start_date, end_date):
    '''
    Returns a dataframe of traffic counts for an entire set of sites
            Parameters:
                    sites_df (dataframe): The sites we want to query, has the same columns as get_quality_area function's output
                    start_date, end_date (str): Strings of the start and end dates of our search ddmmyy
            Returns:
                    report_df (dataframe): Report of traffic counts for the sites
    '''
    # Get the reports on the site
    train_reports =  collections.defaultdict(str)
    # Go through all the site ids and get reports
    print("sites_df: {}".format(sites_df))
    for site_id in tqdm(sites_df.Id):
        report, header = get_site_report(site_id, start_date, end_date)
        report['site_id'] = site_id
        train_reports[site_id] = report

    # Combine reports into one df
    report_df = pd.concat(list(train_reports.values()), ignore_index=True)
    return report_df

def clean_report(report_df):
    '''
    Cleans the traffic count report with a few key steps:
    1. Format the column names and remove redundant columns
    2. Converts the count columns into intergers
    3. Remove rows with blank data
    4. Remove rows that only report one value (zero)
    5. Add a timestamp column to the report

            Parameters:
                    report_df (dataframe): The report of traffic count data outputted by get_reports_from_sites_df

            Returns:
                    clean_report_df (datafrane): The cleaned report
    '''
    # Step 1.
    clean_col_names = [
        'site_name',
        'report_date',
        'time_period_ending',
        'time_interval',
        '0-520cm',
        '521-660cm',
        '661-1160cm',
        '1160+cm',
        '0-10mph',
        '11-15mph',
        '16-20mph',
        '21-25mph',
        '26-30mph',
        '31-35mph',
        '36-40mph',
        '41-45mph',
        '46-50mph',
        '51-55mph',
        '56-60mph',
        '61-70mph',
        '71-80mph',
        '80+mph',
        'avg_mph',
        'total_volume',
        'site_id']
    report_df.columns = clean_col_names
    clean_cols = [
         'site_name',
         'site_id',
         'report_date',
         'time_period_ending',
         'time_interval',
         '0-520cm',
         '521-660cm',
         '661-1160cm',
         '1160+cm',
         'avg_mph',
         'total_volume']
    clean_report_df = report_df[clean_cols]

    # Steps 2., 3., 4.
    interger_cols = [
         '0-520cm',
         '521-660cm',
         '661-1160cm',
         '1160+cm',
         'total_volume']
    def remove_rows(df):
        df = df.loc[df['total_volume'] != '']  # Remove empty rows
        x = df.groupby('site_id')['total_volume'].nunique()
        zero_sites = list(x[x==1].index)  # Remove sites where the volume is always zero
        df = df.loc[~df.site_id.isin(zero_sites)]
        df[interger_cols] = df[interger_cols].astype('int32')
        return df
    clean_report_df = remove_rows(clean_report_df)

    # Step 5.
    def get_timestamp(row):
        year, month,day = row['report_date'].split('T')[0].split('-')
        hour, minute, second = row['time_period_ending'].split(':')
        return datetime.datetime(int(year),int(month),int(day), int(hour), int(minute))

    clean_report_df['timestamp'] = clean_report_df.apply(get_timestamp,axis=1)
    return clean_report_df

# Function used to normalsise the count data
def normalise(clean_report):
    interger_cols = ['0-520cm', '521-660cm', '661-1160cm', '1160+cm', 'total_volume']
    for name in interger_cols:
        new_name = f"{name}_normalised"
        # for ever row in the report present the row's site id's mean volume
        mean = clean_report.groupby('site_id')[name].transform("mean")
        # normalise
        clean_report.loc[:, new_name] = clean_report[name] / mean
        # filter so we don't have rows with a small mean which causes a pole
    return clean_report[mean>1]

# A pipeline of stages for downloading and normalising reporting.
def download_clean_pipeline(start_date, end_date, site_name, max_lat, max_long, min_lat, min_long, quality_threshold = 90, normalise=True):
    print('Producing clean report df using following parameters:')
    print('------------------------------------------')
    print('start_date: {}'.format(start_date))
    print('end_date: {}'.format(end_date))
    print('max_lat: {}'.format(max_lat))
    print('max_long: {}'.format(max_long))
    print('min_lat: {}'.format(min_lat))
    print('min_long: {}'.format(min_long))
    print('quality_threshold: {}'.format(quality_threshold))
    print('normalise: {}'.format(normalise))
    print('------------------------------------------')

    # Get the quality data
    sites_df = get_quality_area(sites, site_name, max_lat, max_long, min_lat, min_long, start_date, end_date, quality_threshold)
    # Download the report
    report_df = get_reports_from_sites_df(sites_df, start_date, end_date)
    # Clean the report
    clean_report_df = clean_report(report_df)
    # Normalsie the report
    if normalise:
        clean_report_df_norm = normalise(clean_report_df)
        return clean_report_df_norm
    else:
        return clean_report_df

"""## Clean Report"""

MAX_LAT, MAX_LONG, MIN_LAT, MIN_LONG = 0, 0, 0, 0 # not used

def trafficapi(site_name, year, aadt_path):

    start_date = '0101'+year
    end_date = '3112'+year

    clean_report_df = download_clean_pipeline(start_date, end_date, site_name, MAX_LAT, MAX_LONG, MIN_LAT, MIN_LONG,  QUALITY_THRESHOLD, normalise=NORMALISE_DF)

    # Groupby site_id and report_date
    site_dfs = [site_day for _, site_day in clean_report_df.groupby(['site_id'])]

    df_aadt_final(site_dfs, START_DATE=start_date, no_days=NUMBER_OF_DAYS, save=True, AADT_PATH=aadt_path)

    return True