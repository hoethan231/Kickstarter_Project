''' 
    Cleaned and engineered Kickstarter dataset for modeling.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

filepath = "/Users/prathamsaxena/Downloads/SJSU/CMPE 188/ML Code/ML Data/kickstarter_data_full.csv"
kickstarter = pd.read_csv(filepath, low_memory=False)

# Select only launch-time-safe features
selected_columns = ['goal', 'state', 'country', 'static_usd_rate', 'category', 'created_at_yr',
                    'create_to_launch_days', 'launch_to_deadline_days', 'blurb_len_clean', 'SuccessfulBool']

kickstarter = kickstarter[selected_columns].dropna()

# Convert goal to USD
kickstarter['goal'] = kickstarter['goal'] * kickstarter['static_usd_rate']
kickstarter = kickstarter.drop('static_usd_rate', axis=1)

# Encode categorical columns
to_encode = ['category', 'country']
encoder = LabelEncoder()
for col in to_encode:
    kickstarter[col] = encoder.fit_transform(kickstarter[col])

# Remove 'live' projects (undecided outcome)
kickstarter = kickstarter[kickstarter['state'] != 'live']
kickstarter = kickstarter.drop('state', axis=1)

# === Feature Engineering ===

# Log of goal to reduce skew
kickstarter['goal_log'] = np.log1p(kickstarter['goal'])

# Goal per day (project ambition)
kickstarter['goal_per_day'] = kickstarter['goal'] / (kickstarter['launch_to_deadline_days'] + 1)

# Is US-based (could impact exposure, success)
kickstarter['is_us'] = (kickstarter['country'] == encoder.transform(['US'])[0]).astype(int)

# Drop original 'goal' to avoid collinearity (optional)
kickstarter = kickstarter.drop('goal', axis=1)

# Shuffle data
kickstarter = kickstarter.sample(frac=1, random_state=42).reset_index(drop=True)

# Train/target split
X = kickstarter.drop(['SuccessfulBool'], axis=1)
Y = kickstarter['SuccessfulBool']

# Optional: Raw version (unused)
rawData = pd.read_csv(filepath, low_memory=False)

toDrop = ["Unnamed: 0", "id", "photo", "name", "blurb", "pledged", "state", "slug", "disable_communication", 
          "currency_symbol", "currency_trailing_code", "deadline", "state_changed_at", "created_at", "launched_at", 
          "staff_pick", "backers_count", "usd_pledged", "creator", "location", "profile", "spotlight", "urls", 
          "source_url", "friends", "is_starred", "is_backing", "permissions", "create_to_launch", "launch_to_deadline", 
          "launch_to_state_change"]

toEncode = ["country", "currency", "category", "deadline_weekday", "state_changed_at_weekday", "created_at_weekday",
            "launched_at_weekday"]

rawDF = rawData.drop(columns=toDrop, axis=1).dropna()
for col in toEncode:
    rawDF[col] = encoder.fit_transform(rawDF[col])

x = rawDF.drop(["SuccessfulBool"], axis=1)
y = pd.DataFrame(rawDF["SuccessfulBool"])
