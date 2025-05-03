''' 
This file will contain conclusions made from my own EDA and work from other individuals compiled here.

The dataset contains attributes that relate to whether a kickstarter failed or succeeded. I will be looking 
into the kickstarter_data_full and analyze the entire data provided by the dataset. I think the features 
.csv file extracts only all the related features for the data set. Nonetheless, I want to look into all the
attributes from the dataset and make my own conclusions. 


'''

import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Tools for preprocessing 
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn.decomposition import PCA 

# Extracting the contain from the .csv file into a Pandas Dataframe
filepath = "C:/Users/Admin/OneDrive/Desktop/Vs/Kickstarter_Project/dataset/kickstarter_data_full.csv"
kickstarter = pd.read_csv(filepath)

'''
    I am going to clear out some of the following columns seen below.
    I will remove ID later, I will check if there are duplicate IDs
    that we should remove. 
    
    Context: 'Blurb' is a short description of the kickstarter; basically tells 
    what the kickerstart is about.
    
    We will need to enumerate the 'state' of the kickerstarter:
        successful
        failed
        cancelled
        live
        suspended
        
    Since the state 'live' is inbetween the failed and successful portions, we can consider kickerstarters in
    'live' state to be failed. So if we create a binary classifier, then we will consider the state 'successful' versus
    all the rest of the states. To simplify our classifier for the project requirements and ourselves, we will do only 
    this one versus all binary classifier. 
    
    The 'slug' state I am pretty sure it is just the event name where the kickerstarter started or something along the 
    lines like that. We can remove that, since they are all unique to themselves. 
    
'''

# Still working in progress... roommate sleeping :( # haha no worries 
print(kickstarter.head())
print(kickstarter.shape)
print(kickstarter.columns)

toDrop = ["Unnamed: 0", "id", "photo", "name", "blurb", "slug", "creator", "location", 
        "profile", "urls", "source_url", "friends", "is_starred", "is_backing", "permissions", 
        "name_len", "name_len_clean", "blurb_len", "blurb_len_clean", "deadline", 
        "state_changed_at", "created_at", "launched_at", "create_to_launch", "launch_to_deadline", 
        "launch_to_state_change", ]

toEncode = ["state", "disable_communication", "country", "currency", "currency_symbol", 
            "currency_trailing_code", "staff_pick", "category", "spotlight", "deadline_weekday", 
            "state_changed_at_weekday", "created_at_weekday", "launched_at_weekday"]

'''
    Catergorical columns like `id`, `photo`, and `name` are dropped because they present unique stings 
    that have no correlation with other other rows meaning that if we encoded them, we would get 20632 
    unique ids. 
    
    There are also some that seem completly useless like `name_len`. 
    
    Columns with dates and other useful information like `deadline`, `created_at`, `launched_at` could 
    be filtered into a new integer column for use. Thankfully the dataset provides these for example 
    splitting the string column `deadline` into an int `deadline_month`, `deadline_year`, and `deadline_day`.
''' 

clean_kickstarter = kickstarter.drop(columns=toDrop, axis=1)
clean_kickstarter.info()
clean_kickstarter.isnull().sum()

'''
    We can see that some coloumns like `catergory` have empty values, we shoudl remove that before any transformations
'''

clean_kickstarter = clean_kickstarter.dropna()

'''
    After dealing with NaN values, we can begin transforming data.
'''

encoder = LabelEncoder()
for col in toEncode:
    clean_kickstarter[col] = encoder.fit_transform(clean_kickstarter[col])
    
clean_kickstarter.hist(figsize=(15, 20), layout=(11, 4))