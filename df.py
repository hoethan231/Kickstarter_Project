''' 
    This file is AFTER the eda process where we jump straight into the cleaned df in order to quickly 
    export into our models later. To see our step by step thought process, checkout eda.py. 
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

filepath = "E:/vs/Kickstarter_Project/dataset/kickstarter_data_full.csv"
kickstarter = pd.read_csv(filepath)

toDrop = ["Unnamed: 0", "id", "photo", "name", "blurb", "slug", "creator", "location", 
        "profile", "urls", "source_url", "friends", "is_starred", "is_backing", "permissions", 
        "name_len", "name_len_clean", "blurb_len", "blurb_len_clean", "deadline", 
        "state_changed_at", "created_at", "launched_at", "create_to_launch", "launch_to_deadline", 
        "launch_to_state_change", "currency_symbol", "deadline_weekday", "disable_communication", 
        "static_usd_rate"]

toEncode = ["state", "country", "currency", 
            "currency_trailing_code", "staff_pick", "category", "spotlight", 
            "state_changed_at_weekday", "created_at_weekday", "launched_at_weekday"]


clean_kickstarter = kickstarter.drop(columns=toDrop, axis=1)
clean_kickstarter = clean_kickstarter.dropna()

encoder = LabelEncoder()
for col in toEncode:
    clean_kickstarter[col] = encoder.fit_transform(clean_kickstarter[col])
    
x = clean_kickstarter.drop(["SuccessfulBool"], axis=1)
y = pd.DataFrame(clean_kickstarter["SuccessfulBool"])

df_srqt_log = clean_kickstarter
for col in x.columns:
    df_srqt_log[col] = np.log2(np.sqrt(df_srqt_log[col]))
df_srqt_log.replace(-np.inf, 0, inplace=True)

df = df_srqt_log
X = df.drop(["SuccessfulBool"], axis=1)
Y = pd.DataFrame(df["SuccessfulBool"])