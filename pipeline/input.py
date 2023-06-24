import pandas as pd

def extract_time_date(time_date):
    
    hour, date = time_date.split(':')
    day, month = date.split('/')

    hour = int(hour)
    day = int(day)
    month = int(month)

    print("Hour:", hour)
    print("Day:", day)
    print("Month:", month)

    return hour, day, month


def convert_site_name(site_name):
    site_name = site_name.replace('/', '_')

    return site_name


def save_args(SITE_NAME, LINK_LENGTH, LINK_LENGTH_DIR, TIME_DATE, TIME_DATE_DIR):

    SITE_NAME = convert_site_name(site_name=SITE_NAME)

    # save link length
    df = pd.DataFrame({'site_name': [SITE_NAME], 'link_length': [LINK_LENGTH]})
    df.to_csv(LINK_LENGTH_DIR+'link_length_'+SITE_NAME+'.csv')

    # save time date
    HOUR, DAY, MONTH = extract_time_date(TIME_DATE)
    df = pd.DataFrame({'site_name': [SITE_NAME], 'day': [DAY], 'month': [MONTH], 'hour': [HOUR]})
    df.to_csv(TIME_DATE_DIR+'time_'+SITE_NAME+'.csv')