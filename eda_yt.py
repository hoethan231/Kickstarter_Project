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
from sklearn.preprocessing import StandardScaler, Normalizer 
from sklearn.decomposition import PCA 

# Extracting the contain from the .csv file into a Pandas Dataframe
filepath = "/Users/yyaatt/Desktop/CMPE188/Final-Project/kickstarter_data_full.csv"
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

# Viewing all the columns 
print(kickstarter.columns)

# Still working in progress... roommate sleeping :(
kickstarter = kickstarter.drop(['Unnamed: 0', 'photo', 'name', 'blurb', 'slug'], axis=1)