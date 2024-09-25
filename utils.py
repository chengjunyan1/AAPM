import yaml
import os
import json
from datetime import datetime, timedelta
import concurrent.futures
from dateutil.relativedelta import relativedelta
import pandas as pd


pjoin=os.path.join
pexists=os.path.exists


def load_yaml(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def save_json(path,jsf):
    with open(path, 'w') as f:
        json.dump(jsf,f)

def load_json(path, default={}):
    if os.path.exists(path):
        with open(path, 'r') as f:
            jsf=json.load(f)
        return jsf
    else: 
        if default is None: raise FileNotFoundError(path)
        else: return default

def load_jsons_parallel(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    result_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_json, os.path.join(directory, file)): file for file in json_files}
        for future in concurrent.futures.as_completed(futures):
            file = futures[future]
            try:
                data = future.result()
                result_list.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return result_list

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def date_add(original_date_str,d=0,w=0,m=0,y=0,h=0,fmt="%Y-%m-%d"):
    original_date = datetime.strptime(original_date_str, fmt)
    w+=52*y
    result_date = original_date + relativedelta(days=d,weeks=w,months=m,hours=h)
    result_date_str = result_date.strftime(fmt)
    return result_date_str  # Output: 2022-02-15


def dt2str(dt:pd.Timestamp, fmt="%Y-%m-%d %H:%M:%S%z"): # time like 2016-11-04 00:00:00+00:00
    return dt.strftime(fmt)

def str2dt(dtstr,fmt="%Y-%m-%d %H:%M:%S%z"): # reverse above
    return datetime.strptime(dtstr, fmt)


def replace_forbidden_chars(file_name):
    file_name=file_name.replace(' ','_')
    forbidden_chars = r'<>:"/\|?*'
    translation_table = str.maketrans(forbidden_chars, '_' * len(forbidden_chars))
    return file_name.translate(translation_table).lower()


