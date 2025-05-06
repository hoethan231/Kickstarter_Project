''' 
    This file is AFTER the eda process where we jump straight into the cleaned df in order to quickly 
    export into our models later. To see our step by step thought process, checkout eda.py. 
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

filepath = "/Users/yyaatt/Desktop/CMPE188/Final-Project/kickstarter_data_full.csv"
kickstarter = pd.read_csv(filepath)

selected_columns = ['goal', 'country', 'static_usd_rate', 'staff_pick', 'backers_count', 'usd_pledged', 'category', 'spotlight', 
                   'created_at_yr', 'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days', 'SuccessfulBool']

kickstarter = kickstarter[selected_columns]

kickstarter = kickstarter.dropna()

kickstarter['goal'] = kickstarter['goal'] * kickstarter['static_usd_rate']

# We don't need that column, just to convert the goals data
kickstarter = kickstarter.drop('static_usd_rate', axis=1)

# Encoding category and country
to_encode = ['category', 'country']
encoder = LabelEncoder()
for col in to_encode:
    kickstarter[col] = encoder.fit_transform(kickstarter[col])
    
X = kickstarter.drop(['SuccessfulBool'], axis=1)
Y = kickstarter['SuccessfulBool']