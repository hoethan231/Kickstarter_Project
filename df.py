''' 
    This file is AFTER the eda process where we jump straight into the cleaned df in order to quickly 
    export into our models later. To see our step by step thought process, checkout eda.py. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

filepath = "/Users/yyaatt/Desktop/CMPE188/Final-Project/kickstarter_data_full.csv"
kickstarter = pd.read_csv(filepath)

selected_columns = ['goal', 'state', 'country', 'static_usd_rate', 'category', 'created_at_yr', 'create_to_launch_days', 'launch_to_deadline_days', 'blurb_len_clean', 'SuccessfulBool']

kickstarter = kickstarter[selected_columns]

kickstarter = kickstarter.dropna()

kickstarter['goal'] = kickstarter['goal'] * kickstarter['static_usd_rate']
# We don't need that column, just to convert the goals data
kickstarter = kickstarter.drop('static_usd_rate', axis=1)

to_encode = ['category', 'country']
encoder = LabelEncoder()
for col in to_encode:
    kickstarter[col] = encoder.fit_transform(kickstarter[col])

# One group mentioned that live state is inbetween failed and successful states. Since we cannot differentiate 
# between either live state is in failed or successful, we will remove all rows that contain live state. 
kickstarter = kickstarter[kickstarter['state'] != 'live']
kickstarter = kickstarter.drop('state', axis=1)

# One group mentioned that some of these columns are sorted, so we should randomize the dataframe
kickstarter = kickstarter.sample(frac=1, random_state=42).reset_index(drop=True)

X = kickstarter.drop(['SuccessfulBool'], axis=1)
Y = kickstarter['SuccessfulBool']

# x and y below will essentially be the raw version of kickstarter before EDA
# The above X and Y are the cleaner versions and should be used instead.

rawData = pd.read_csv(filepath)


toDrop = ["Unnamed: 0", "id", "photo", "name", "blurb", "pledged", "state", "slug", "disable_communication", 
          "currency_symbol", "currency_trailing_code", "deadline", "state_changed_at", "created_at", "launched_at", 
          "staff_pick", "backers_count", "usd_pledged", "creator", "location", "profile", "spotlight", "urls", 
          "source_url", "friends", "is_starred", "is_backing", "permissions", "create_to_launch", "launch_to_dealine", 
          "launch_to_state_change"]

toEncode = ["country", "currency", "category", "deadline_weekday", "state_chnaged_at_weekday", "created_at_weekday",
            "launched_at_weekday"]

rawDF = rawData.drop(columns=toDrop, axis=1)
rawDF = rawDF.dropna()

encoder = LabelEncoder()
for col in toEncode:
    rawDF[col] = encoder.fit_transform(rawDF[col])
    
x = rawDF.drop(["SuccessfulBool"], axis=1)
y = pd.DataFrame(rawDF["SuccessfulBool"])