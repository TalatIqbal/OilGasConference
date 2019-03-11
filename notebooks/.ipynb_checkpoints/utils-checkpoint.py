
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import os
import ntpath
import pickle as pkl
import xlrd
import time
import string
import os
import glob
import math

from os import listdir
from os.path import isfile, join


# ### File Operations

# In[21]:


def remove_file_in_folder(folder_path):

    files = glob.glob(folder_path + '*')
    for f in files:
        os.remove(f)


# ### String operations

# In[22]:


def remove_punctuation(x):
    table = str.maketrans({key: None for key in string.punctuation})
    return x.translate(table)


# In[23]:


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# In[24]:


def get_time(dt_str):
    dt_str = dt_str.strip()
    dtobj = datetime.strptime(dt_str, '%m/%d/%Y %I:%M:%S %p')
    return dtobj


# In[25]:


def parse(txt):
    '''
    @{PIPoint=SCTM:22GTWY_E403:FALE22E23SP.PNT; Value=60; Timestamp=12/30/2017 11:48:05 PM}
    '''
    pi_point, val, time  = None, None, None
    delimiter = ';'
    sub_delimiter = '='
    
    txt = txt[txt.find('{')+1:txt.find('}')]    
    parsed_vals = txt.split(';')
    
    if len(parsed_vals) >= 3:
        pi_point = parsed_vals[0].split(sub_delimiter)[1]
    
        if pi_point is not None:
            values = parsed_vals[1].split(sub_delimiter)
            if len(values) >= 2:
                val = values[1]
                if is_number(val):
                    val = float(val)
                else:
                    val = None
            else:
                val = None

            time_vals = parsed_vals[2].split(sub_delimiter)
            if len(time_vals) >= 2:
                time = time_vals[1]
                time = get_time(time)
            else:
                return None, None, None

    if pi_point is not None:
        pi_point = pi_point.replace('SCTM:', '')
    
    return pi_point, val, time    


# In[26]:


def longestSubstringFinder(string1, string2):
    '''
    Code from https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings    
    '''
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return answer


# ### Reading Data

# In[3]:


def read_data_withfeature(input_path, print_debug = False):
    
    df_features = {}
    
    if os.path.isdir(input_path):
        input_files = [f for f in listdir(input_path) if (isfile(join(input_path, f))) and ((f.endswith('.pkl')) or (f.endswith('.csv')))]
    elif os.path.isfile(input_path):
        input_files = input_path
    
    if print_debug:
        print('Number of files found in %s is %d ' % (input_path, len(input_files)))
    
    for input_file in input_files:
        # feature,_ = os.path.splitext(input_file)
        input_file = input_path + input_file  
        
        with open(input_file, 'rb') as f:
            df = pkl.load(f)
            unq_features = np.unique(df['feature'])
            print(input_file, unq_features)
            if len(unq_features) > 0:
                if len(unq_features) > 1:
                    if print_debug:
                        print('There are %d features in file %s' % (len(unq_features), input_file))
                    continue            
                feature = unq_features[0]            
                df_features[feature]= df
    
    if print_debug:
        print('Number of features extracted from %d files is %d ' % (len(input_files), len(df_features)))
    
    return df_features


# In[2]:


def read_data(input_path, print_debug = False):
    
    df_features = {}
    
    if os.path.isdir(input_path):
        input_files = [f for f in listdir(input_path) if (isfile(join(input_path, f))) and ((f.endswith('.pkl')) or (f.endswith('.csv')))]
    elif os.path.isfile(input_path):
        input_files = input_path
    
    if print_debug:
        print('Number of files found in %s is %d ' % (input_path, len(input_files)))
    
    for input_file in input_files:
        
        feature,_ = os.path.splitext(input_file)
        input_file = input_path + input_file  
        feature = feature.replace('-', ':')
                
        with open(input_file, 'rb') as f:
            df = pkl.load(f)
            df_features[feature] = df
                
    if print_debug:
        print('Number of features extracted from %d files is %d ' % (len(input_files), len(df_features)))
    
    return df_features


# ### Generate Statistics Table

# In[28]:


def generate_stats_df(df_features):

    df_stats = pd.DataFrame(columns=['feature', 'total_count', 'missing_val_count', 'min_date', 'max_date', 'max_val', 'min_val', 'variance', 'std', 'mean_val', 'median_val', 'kurt', 'skew'])
    idx = 0

    for feature, df in df_features.items(): 
        df_stats.loc[idx] = [feature, len(df), len(df)-df['val'].count(), df['datetime'].min(), df['datetime'].max(), df['val'].max(), df['val'].min(), df['val'].var(), df['val'].std(), df['val'].mean(), df['val'].median(), df['val'].kurt(), df['val'].skew()]
        idx += 1
        
    # Get the percentage missing values
    df_stats['perc_missing'] = df_stats['missing_val_count']/df_stats['total_count'] * 100
    return df_stats


# ### Remove features with substantial missing values

# In[29]:


def remove_missing_features(df_features, value_col, max_missing_vals_pcnt):
    features = list(df_features.keys())
    for feature in features:
        df = df_features[feature]
        num_rows = len(df)
        count = df[value_col].count()
        percent_missing = (num_rows-count)/num_rows * 100
        # print(feature, percent_missing)
        if percent_missing > max_missing_vals_pcnt:
            del df_features[feature]
            # print('-------------------')
            
    return df_features


# ### Dateime <--> Integer Functions

# In[30]:


def get_seconds_after(current_date, base_date):
    
    base_ts = time.mktime(base_date.timetuple()) # Converting to Unix timestamp
    current_ts = time.mktime(current_date.timetuple())
    time_diff = round((current_ts - base_ts))
    
    return time_diff


# In[31]:


def get_minutes_after(current_date, base_date):
        
    base_ts = time.mktime(base_date.timetuple()) # Converting to Unix timestamp
    current_ts = time.mktime(current_date.timetuple())
    time_diff = round((current_ts - base_ts) / 60.0) + 1
    
    return time_diff


# In[32]:


def get_hours_after(current_date, base_date):
    
    base_ts = time.mktime(base_date.timetuple()) # Converting to Unix timestamp
    current_ts = time.mktime(current_date.timetuple())
    time_diff = round((current_ts - base_ts) / 60.0 / 60.0) + 1
    
    return time_diff


# In[33]:


def get_days_after(current_date, base_date):
    
    base_ts = time.mktime(base_date.timetuple()) # Converting to Unix timestamp
    current_ts = time.mktime(current_date.timetuple())
    time_diff = round((current_ts - base_ts) / 60.0 / 60.0 / 24) + 1
    
    return time_diff


# In[34]:


def get_months_after(current_date, base_date):    
    time_diff = ((current_date.year - base_date.year) * 12) + current_date.month - base_date.month + 1    
    return time_diff


# In[35]:


def get_years_after(current_date, base_date):    
    time_diff = (current_date.year - base_date.year) + 1 
    return time_diff


# In[36]:


def get_time_from_minutes(time_in_mins, base_date):    
    new_date = base_date + timedelta(minutes = time_in_mins)        
    return new_date


# In[37]:


def get_granulairty_function(granularity, current_date, base_date):
    
    if granularity == 'sec':
        return get_seconds_after(current_date, base_date)
    elif granularity == 'min':
        return get_minutes_after(current_date, base_date)
    elif granularity == 'hr':
        return get_hours_after(current_date, base_date)
    elif granularity == 'day':
        return get_days_after(current_date, base_date)
    elif granularity == 'mon':
        return get_months_after(current_date, base_date)
    elif granularity == 'yr':
        return get_years_after(current_date, base_date)
    
    return get_minutes_after(current_date, base_date) # Default return function is for minutes


# ### Data Aggregation for duplicate timestamps

# In[1]:


def agg_data_dup_timestamps(df_features, 
                            feature_set, 
                            time_granularity, 
                            time_col, 
                            value_col, 
                            time_gran_col, 
                            base_date):
    
    df_features_e = {}
    
    if len(feature_set) == 0:
        feature_set = list(df_features.keys())
        
    for feature in feature_set:
        
        print(feature + ' -- Started ', end='')
        
        df = df_features[feature].copy()
        len_df_before = len(df)
        
        df.drop(columns=['feature'], inplace=True) # Drop the feature column as its redundant

        # Drop duplicates
        df.drop_duplicates(inplace=True)
        df.sort_values(by=[time_col], inplace=True, ascending=True)

        # Compute the granularity
        df[time_gran_col] = df[time_col].apply(lambda x:get_granulairty_function(time_granularity, x, base_date))

        # Average if there are more readings within the same granularity level
        df = df.dropna(subset=[value_col])
        df = df.drop(columns=[time_col])

        # Convert the value column to numeric to help in aggregation of duplicate timestamps
        df[value_col] = pd.to_numeric(df[value_col])

        # AGGREGATE the duplicate timestamps - take the mean
        df_g = df[[time_gran_col, value_col]].groupby(time_gran_col).mean()
        df = df_g.reset_index(level=0, inplace=False)
        len_df_after = len(df)
        
        print('Ended - Aggregated %d rows to %d rows' % (len_df_before, len_df_after))
        
        
        df_features_e[feature] = df 
        
        
    return df_features_e


# ### Writing down files

# In[44]:


def write_feature_dict(dir_path, df_features, remove_existing=True):
    
    if remove_existing:
        # Remove the files from the directory
        remove_file_in_folder(dir_path)

    # Write the pickle files to the folder
    for feature in df_features.keys():
        df = df_features[feature].copy()
        
        fname = feature.replace(':', '-')

        pkl_file = dir_path + fname + '.pkl'
        print('Writing to file ', pkl_file, df.shape, '[', feature, ']')
        
        with open(pkl_file, 'wb') as f:
            pkl.dump(df, f, protocol=pkl.HIGHEST_PROTOCOL)
            


# ### Generate Master Dataframe for time

# In[40]:


from datetime import datetime
def generate_master_df(time_granularity,
                       time_gran_col,
                       base_date,
                       end_date,
                       print_debug = False):
    '''
    Generates a master dataframe
    Dataframe will have an integer column that denotes x minutes have passed after the base_date
    granulaity - can take one of the following - 'sec' (seconds), min ' (minutes), 'hr' (hour), 
                'day' (day), 'mon' (month), 'yr' (year)
    base_date = date of reference since which the unit of time is computed
    '''
    
    if print_debug:
        print('Granularity is', time_granularity, '\tStart Date = ', base_date, '\tEnd Date = ', end_date)
            
    max_td = get_granulairty_function(time_granularity, end_date, base_date)
    
    df_master = pd.DataFrame(columns=[time_gran_col])    
    df_master[time_gran_col] = [i for i in range(1, max_td+1)]
    
    if print_debug:
        print('Shape of the master dataframe is ', df_master.shape, 'with columns ', df_master.columns.values)
    
    return df_master


# ### Normalization

# In[41]:


def scale_val(val, min_val, max_val):
    if val is not None:
        return (val-min_val)/(max_val-min_val + 1e-7)
    return None

def lcl_divmul(val, div_by, mul_by):
    val = math.floor(val/div_by)
    val = val * mul_by
    return val

