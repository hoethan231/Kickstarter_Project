''' 
    This file is AFTER the eda process where we jump straight into the cleaned df in order to quickly 
    export into our models later. To see our step by step thought process, checkout eda.py. 
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

filepath = "/Users/yyaatt/Desktop/CMPE188/Final-Project/kickstarter_data_full.csv"
kickstarter = pd.read_csv(filepath)

# selected_columns = ['goal', 'state', 'country', 'static_usd_rate', 'staff_pick', 'backers_count', 'usd_pledged', 'category', 'spotlight', 
#                     'created_at_yr', 'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days', 'name_len_clean', 
#                     'blurb_len_clean', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday', 'USorGB', 
#                     'TOPCOUNTRY', 'LaunchedTuesday', 'DeadlineWeekend', 'SuccessfulBool']
selected_columns = ['goal', 'state', 'country', 'static_usd_rate', 'category', 'created_at_yr', 'create_to_launch_days', 'launch_to_deadline_days', 'blurb_len_clean', 'SuccessfulBool']

kickstarter = kickstarter[selected_columns]

kickstarter = kickstarter.dropna()

kickstarter['goal'] = kickstarter['goal'] * kickstarter['static_usd_rate']

# We don't need that column, just to convert the goals data
kickstarter = kickstarter.drop('static_usd_rate', axis=1)

# Encoding category and country
to_encode = ['category', 'country']
# to_encode = ['category', 'country', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday']
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

toDrop = ["Unnamed: 0", "id", "photo", "name", "blurb", "slug", "creator", "location", 
        "profile", "urls", "source_url", "friends", "is_starred", "is_backing", "permissions", 
        "name_len", "name_len_clean", "blurb_len", "blurb_len_clean", "deadline", 
        "state_changed_at", "created_at", "launched_at", "create_to_launch", "launch_to_deadline", 
        "launch_to_state_change", "state", "spotlight"]

toEncode = ["disable_communication", "country", "currency", "currency_symbol", 
            "currency_trailing_code", "staff_pick", "category", "deadline_weekday", 
            "state_changed_at_weekday", "created_at_weekday", "launched_at_weekday"]

rawDF = rawData.drop(columns=toDrop, axis=1)
rawDF = rawDF.dropna()

encoder = LabelEncoder()
for col in toEncode:
    rawDF[col] = encoder.fit_transform(rawDF[col])
    
x = rawDF.drop(["SuccessfulBool"], axis=1)
y = pd.DataFrame(rawDF["SuccessfulBool"])