import pandas as pd
import shutil

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


def process_image_id(LA, site_name):
    site_name = site_name.replace('/', '_')

    name = LA.lower()+'_'+site_name.lower()

    return name

def copy_file(source_path, target_path):
    try:
        shutil.copy2(source_path, target_path)
        print(f"File copied successfully from {source_path} to {target_path}")
    except FileNotFoundError:
        print("Error: File not found.")
    except IsADirectoryError:
        print("Error: Source path is a directory.")
    except shutil.SameFileError:
        print("Error: Source and target paths are the same.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



def save_args(LA, SITE_NAME, LINK_LENGTH, LINK_LENGTH_DIR, TRUE_SPEED, TRUE_SPEED_DIR, TIME_DATE, TIME_DATE_DIR, SAT_IMAGE_PATH, IMAGE_DIR):

    SITE_NAME = process_image_id(LA=LA, site_name=SITE_NAME)

    # save link length
    df = pd.DataFrame({'image_id': [SITE_NAME], 'link_length': [LINK_LENGTH]})
    df.to_csv(LINK_LENGTH_DIR+'link_length_'+SITE_NAME+'.csv')

    # save time date
    HOUR, DAY, MONTH = extract_time_date(TIME_DATE)
    df = pd.DataFrame({'image_id': [SITE_NAME], 'day': [DAY], 'month': [MONTH], 'hour': [HOUR]})
    df.to_csv(TIME_DATE_DIR+'time_'+SITE_NAME+'.csv')
    

    # save speed
    df = pd.DataFrame({'image_id': [SITE_NAME], 'avg_mph': [TRUE_SPEED]})
    df.to_csv(TRUE_SPEED_DIR+'avg_mph_'+SITE_NAME+'.csv')

    # copying satellite image
    copy_file(SAT_IMAGE_PATH, IMAGE_DIR)