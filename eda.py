#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Tools for preprocessing 
from sklearn.preprocessing import StandardScaler, Normalizer 
from sklearn.preprocessing import LabelEncoder
# from sklearn.decomposition import PCA 


# In[ ]:


# Extracting the contain from the .csv file into a Pandas Dataframe

''' https://www.kaggle.com/datasets/sripaadsrinivasan/kickstarter-campaigns-dataset '''

filepath = "/Users/yyaatt/Desktop/CMPE188/Final-Project/kickstarter_data_full.csv"
kickstarter = pd.read_csv(filepath)


# In[ ]:


# Viewing all the columns 
kickstarter.columns


# In[ ]:


"""
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

'''
I think we should still include location because location could be a factor is a kickstarter idea is better than 
in one location compared to another. We just have to clean that portion out.

Okay on my first comment about the state of the kickstarter, I think maybe we can include that as a feature or 
something or just to note the demographic of our dataset. So we can understand is our dataset possibly bias on
one state or what. Maybe not use it as a feature for our clasification model but can consider. There is a feature 
called "SuccessfulBool", so we can use that as our OUTPUT in our binary classifier.
"""


# In[ ]:


'''
 0   goal                         20632 non-null  float64
    
        I am assuming that the goal currency is in their local currency, so we may have to convert them, we multiple each goal
        with the static_usd_rate. Therefore, all our currency is in USD rather than in different types of currency.
 
 1   pledged                      20632 non-null  float64
 2   state                        20632 non-null  object 
 3   country                      20632 non-null  object 
 4   currency                     20632 non-null  object 
 5   currency_trailing_code       20632 non-null  bool      
        
        We can prob delete that, we have to look into what that means
    
 6   staff_pick                   20632 non-null  bool   
 7   backers_count                20632 non-null  int64  
 8   static_usd_rate              20632 non-null  float64

        We are definately dropping that because we have the year the kickstarter was created and the pledge amount in USD.
        We will still use it but for like the models used later, we will ignore this model. 

 9   usd_pledged                  20632 non-null  float64
 10  category                     18743 non-null  object 
 
        There is some NaN (10% of data is mislabeled), so I am thinking we should label those NaN to something.
 
 11  spotlight                    20632 non-null  bool   
 12  state_changed_at_weekday     20632 non-null  object 
 
        This is prob useless, but I would want to see if there is any correlation with the "state" feature
 
 13  created_at_weekday           20632 non-null  object 
 
        This goes the same for the weekday and I want to see if there is any relationship between the weekday or something
        and the state of the kickstarter and perhaps if the kickstarter is successful or not. I really doubt there is any
        relationship between them but whatever.
    
 14  launched_at_weekday          20632 non-null  object 
 
        Again, same goes here.
 
 15  deadline_month               20632 non-null  int64  
 
        Maybe consider this and see its relationship of the state of the kickstarter?
 
 16  deadline_day                 20632 non-null  int64  
 
        Maybe consider this and see its relationship of the state of the kickstarter?
 
 17  deadline_yr                  20632 non-null  int64  
 
        Maybe consider this and see its relationship of the state of the kickstarter?
 
 18  deadline_hr                  20632 non-null  int64  
 
        Maybe consider this and see its relationship of the state of the kickstarter? For all the deadlines, I am assuming 
        there are unique deadlines specific for each country that could impact it. 
 
 19  state_changed_at_month       20632 non-null  int64  
 
        Prob this because looking at the histogram, it is basically all even throughout, there is no noticeable pattern considering
        the entire dataset. We could look at each country but I think that is irrelavent to know for our models.
 
 20  state_changed_at_day         20632 non-null  int64  
    
        Extra information. Can remove.
 
 21  state_changed_at_yr          20632 non-null  int64  
 
        Comparing the histogram between this and the year created, they are like one-to-one. I am thinking to drop this because 
        this is extra information we most likely do not need. 
 
 22  state_changed_at_hr          20632 non-null  int64  
 
        Extra information. Can remove. 
 
 23  created_at_month             20632 non-null  int64  
 24  created_at_day               20632 non-null  int64  
 25  created_at_yr                20632 non-null  int64  
 26  created_at_hr                20632 non-null  int64  
 27  launched_at_month            20632 non-null  int64  
 28  launched_at_day              20632 non-null  int64  
 29  launched_at_yr               20632 non-null  int64  
 30  launched_at_hr               20632 non-null  int64  
 31  create_to_launch_days        20632 non-null  int64  
 32  launch_to_deadline_days      20632 non-null  int64  
 33  launch_to_state_change_days  20632 non-null  int64  
 34  SuccessfulBool               20632 non-null  int64  
 35  USorGB                       20632 non-null  int64  
 
       It seems this feature is going by to see if the country of the kickstarter is from the United States or Great Britan.
       As a result, this feature should be type BOOL instead of INT64. Since we already include country as one of our features,
       we can ignore this feature.
       
 36  TOPCOUNTRY                   20632 non-null  int64  
 
       My initial assumption is that TOPCOUNTRY refers to countries such as US or something along like that, so I will have to plot 
       a relationship between country and if it is classified as a top country. TOPCOUNTRY should also be BOOL instead of INT64.
 
 37  LaunchedTuesday              20632 non-null  int64  
 
       I think this legit means that a startup just launched on Tuesday, this is kinda random. This should also be of type BOOL and not INT64.
 
 38  DeadlineWeekend              20632 non-null  int64  
       
       I think this legit means that a kickstarter has a deadline on the weekened. This should also be a type BOOL rather than 
       not INT64. Since we included the other features that already include this like deadline_day. I think we should just ignore these. 
'''


# In[ ]:


features_list = ['goal', 'pledged', 'state', 'country', 'currency', 'currency_trailing_code',
                 'staff_pick', 'backers_count', 'static_usd_rate', 'usd_pledged', 'category',
                 'spotlight', 'state_changed_at_weekday', 'created_at_weekday', 'launched_at_weekday',
                 'deadline_month', 'deadline_day', 'deadline_yr', 'deadline_hr' , 'state_changed_at_month',
                 'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr', 'created_at_month', 
                 'created_at_day', 'created_at_yr', 'created_at_hr', 'launched_at_month', 
                 'launched_at_day', 'launched_at_yr', 'launched_at_hr', 'create_to_launch_days',
                 'launch_to_deadline_days', 'launch_to_state_change_days', 'SuccessfulBool', 'name_len_clean', 'TOPCOUNTRY', 'LaunchedTuesday', 'DeadlineWeekend', 'blurb_len_clean']

'''
After hearing from other groups, we want to add some other features to consider and investigate. Another 
point that other groups has mentioned is that we should investigate characteristics about kickstarters 
initiallally and try to investigate features that may cause a data leakage. 
''' 

additional_features = ['name_len_clean']

# Selecting features that we think are important
kickstarter = kickstarter[features_list]


# In[ ]:


kickstarter = kickstarter.dropna()

# Again, this realy sucks cause we are removing a lot about like 10% of data 
# (dropped ~2000 kickerstarters) which is valuable to us


# In[ ]:


# Displaying the object type of each feature
kickstarter.dtypes


# In[ ]:


# Observing the demographic of our data
plt.figure(figsize=(10, 6))
kickstarter['country'].value_counts().plot(kind='bar')
plt.title('Number of Kickerstarters Per Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

''' 
The dataset does not represent the entire population of kickerstarters with their sample space. They primarly represent US kickerstarters.
As a result, whatever conclusions from our analysis may be a good assumption for US kickstarters, but we cannot say for other kickerstarters
from other countries.
'''


# In[ ]:


# Seeing all of the count for each kickstarter state
plt.figure(figsize=(10, 6))
kickstarter['state'].value_counts().plot(kind='bar')
plt.title('Kickstarter State Count')
plt.ylabel('Count')
plt.ylim(0, 12000)
plt.xticks(rotation=45)
plt.show()

# We can make a cleaner by removing the 'live' state from our dataset, since SuccessfulBool will categorize that as failed.
# Since canceled and suspended states I think can also be considered failed, we will leave those as is. If we are still going by
# with our binary classification. 


# In[ ]:


# Seeing which Kickerstarter was classified at sucessful or not based
plt.figure(figsize=(10, 6))
kickstarter['SuccessfulBool'].value_counts().plot(kind='bar')
plt.title('Ratio of Successful or Not Kickerstarters')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

'''
So the dataset basically simplified the state of the kickerstarter. Anything that isn't successful is placed at not successful. 
For the actual understanding of the dataset, we will use the state of the kickerstarter to gauge a fuller understanding. Afterwards, 
we will use the SuccessfulBool as our output for our classifier models. 
'''


# In[ ]:


# We want to prove if state and SuccessfulBool are the same (outputs) by counting the number of successful states and 
# number of 1's in SuccessfulBool

count_state = kickstarter['state'].value_counts()['successful']
count_success_bool = (kickstarter['SuccessfulBool'] == 1).sum()

print(f"State: {count_state}")
print(f"SuccessfulBool: {count_success_bool}")

# They are exactly the same so we MUST eliminate state as a input for our models


# In[ ]:


# We also want to see the time created versus the success of a kickstarter; again, 
# this could be due to a number of factors such the category the kickstarter falls under

counts = kickstarter.groupby(['created_at_yr', 'state']).size().unstack(fill_value=0)

years_sorted = counts.index.tolist()
states = counts.columns.tolist()
x = np.arange(len(years_sorted))
width = 0.8 / len(states)

# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))
for i, cat in enumerate(states):
    ax.bar(x + i*width, counts[cat], width=width, label=cat)

ax.set_xlabel('Created Year')
ax.set_ylabel('Number of Projects')
ax.set_title('Kickstarter Projects by Created Year and State')
ax.set_xticks(x + width*(len(states)-1)/2)
ax.set_xticklabels(years_sorted, rotation=45)
ax.legend(title='Category')
plt.tight_layout()
plt.show()

''' 
    This tells us that majority of our kickstarters that were sampled were created from 2014 to 2016. It 
    appears that more than half of the startups from that each year within that range succeeded. We count the other
    states other than successful as failed. 
'''


# In[ ]:


# Let's see the relationship of each countries kickerstarted and the state of them
counts = kickstarter.groupby(['country','state']).size().unstack(fill_value=0)
for country in counts.index:
    freqs = counts.loc[country]
    plt.figure()
    plt.bar(freqs.index, freqs.values)
    plt.title(f'Kickstarter State Counts in {country}')
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
'''
Majority of the kickstarters sampled from each country shows sampling more failed startups compared to the others. The next runner up is successful starterups.
As a result, for each country, we can get a better understanding why these kickstarters failed compared to why they succeeded or fall in other states.
'''


# In[ ]:


# I want to see each countries demographic of kickestarters and the number of successful kickstarters for each country
plt.figure(figsize=(10, 6))
kickstarter['category'].value_counts().plot(kind='bar')
plt.title('Number of Kickerstarters Per Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
'''
It's a common theme that majority of the kickerstarters fall into web, hardware, or software kickerstarter. It seems that majority of the kickerstarters that were sampled
are some tech-related kickstarter. There are a few kickstarters that were sampled outside of the tech-related industry (like Silicon Valley tech-related). Conclusions again 
will be narrowed specifically for kickstarters related to tech.

There were some unlabeled categories that was unfortunately dropped. That resulted in about losing 10% of our data, so if we had time, we can go back and correctly label them 
based on information provided by the sample. For an easier time understanding the data, we can look what is already categorized. 
''' 


# In[ ]:


# Knowing the majority of categories and the range of years that the samples primarily came from and majority of 
# the kickstarters are tech related (I will include hardware, software, and web together since that comprised like
# half of our data; gadgets could be a number of things) I will look at the success ratio of those. I will then look
# into the other kickstarter categories.

tech_categories = ['Web', 'Hardware', 'Software', 'Apps', 'Robots']
other_categories = ['Gadgets', 'Plays', 'Wearables', 'Musical', 'Sound', 
                    'Festivals', 'Flight', 'Experimental', 'Immersive', 
                    'Makerspaces Spaces', 'Places', 'Shorts', 'Thrillers',
                    'Webseries', 'Restaurants Blues', 'Academic', 'Comedy']

tech_df = kickstarter[kickstarter['category'].isin(tech_categories)]
other_df = kickstarter[kickstarter['category'].isin(other_categories)]


# In[ ]:


# Shows the group of tech categories and see the success rate for each year that is was created
counts = tech_df.groupby(['created_at_yr', 'state']).size().unstack(fill_value=0)

years_sorted = counts.index.tolist()
states = counts.columns.tolist()
x = np.arange(len(years_sorted))
width = 0.8 / len(states)

# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))
for i, cat in enumerate(states):
    ax.bar(x + i*width, counts[cat], width=width, label=cat)

ax.set_xlabel('Created Year')
ax.set_ylabel('Number of Projects')
ax.set_title('Tech Kickstarter Projects by Created Year and State')
ax.set_xticks(x + width*(len(states)-1)/2)
ax.set_xticklabels(years_sorted, rotation=45)
ax.set_ylim(0, 2500)
ax.legend(title='Category')
plt.tight_layout()
plt.show()


# In[ ]:


# Shows the group of other categories and see the success rate for each year that is was created
counts = other_df.groupby(['created_at_yr', 'state']).size().unstack(fill_value=0)

years_sorted = counts.index.tolist()
states = counts.columns.tolist()
x = np.arange(len(years_sorted))
width = 0.8 / len(states)

# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))
for i, cat in enumerate(states):
    ax.bar(x + i*width, counts[cat], width=width, label=cat)

ax.set_xlabel('Created Year')
ax.set_ylabel('Number of Projects')
ax.set_title('Non-Tech Kickstarter Projects by Created Year and State')
ax.set_xticks(x + width*(len(states)-1)/2)
ax.set_xticklabels(years_sorted, rotation=45)
ax.legend(title='Category')
plt.tight_layout()
plt.show()


# In[ ]:


'''
Again, these two bar plots that is split between what we classified as tech startups and the test indicates that 
there is a good difference between groups to help classify if a startup will fail or not. We can see that from the 
predominate kickstarters that were created from years 2014-2016 sampled, we can see that those who are tech-related 
kickstarters fail more often than succeeded. If we take a look at other kickstarters that aren't tech-related, it seems
that there is about a 60 to 40 ratio of a failed to successful kickstarter. This makes sense because at this range of years 
is when many tech-related kickstarters began and the tech boom became dominate. As a result, there are also multiple 
kickstarters in our dataset and the market for tech-related stuff is becoming noticeable saturated (maybe not a good word);
therefore, we can see a higher number of tech kickstarters fail. If we take a look at other kickstarters, the market maybe more
open for them to succeed. As a result, we can see more of these starts succeed within this range of years.

With that, we have another layer of differences to help classify within our models.

So in our classification models applied later on my guess is that since our data is majority on the kickstarters that were created
from 2014 to 2016 and tech-related ones, we can accurate predict the outcome of startups that fall within those ranges. 
Outside of that, we have little data to ensure accuracy.
'''


# In[ ]:


# A scatterplot displaying the relationship between the number of backers that supported kickstarters
# and how much they raised. A color coordination indicates the kickstarter's state 

for state in kickstarter['state'].unique():
    fig, ax = plt.subplots()
    subset = kickstarter[kickstarter['state'] == state]
    ax.scatter(subset['backers_count'], subset['usd_pledged'])
    if(state == 'successful'):
        ax.set_xlim(0, 15000)
        ax.set_ylim(0, 3000000)
    if(state == 'failed'):
        ax.set_xlim(0, 800)
    ax.set_title(f'Kickestarter State: {state}')
    ax.set_xlabel('Backers Count')
    ax.set_ylabel('USD Pledge')
plt.show()

'''
    Since we are looking at the entire demographic of the kickstarters and not their specific category, it is clear
    that failed kickstarters cluster around having less than 800 backers (people who pay to support their kickstarter)
    and raised less than about $100,000 USD. If we take a look at the successful kickstarters and look at where its primarily
    at, it seems that even despite raising the same amounts and the same number of backers as the failed ones, there are plenty of kickstarters
    that became successful. What is more apparent is that the more backers and money, the chance of succeeding seems to be higher
    due to large amount of kickerstarters seem to be successful at that area. 
    
    So in conclusion from looking primarily at the successful and failed kickstarters (disregarding the category and majority of the kickstarters
    are based in the US), the more money raised and backers will indicate a higher chance of succeeding. However, even having 
    small amounts of backers or minimum money raised, many kickstarters can still succeed. If we remember the number of successful 
    and failed kickstarters in our dataset, it seems that majority of kickstarters that failed is due to a lack of money raised
    and number of supporters. But since this could cary due to the genre the kickstarter is trying to break into, then we cannot 
    say for certain. 
    
    From this, it is clear that we both need the category and amount raised to determine the success of a kickerstarter. The input 
    state is a cheatcode, which we cannot use. 
'''


# In[ ]:


# I am assuming that the amount in the goals column is in their local currency, so I will convert them to USD based on 
# static_usd_rate. That was used to convert pledge to usd_pledge. We should drop pledge since the currecny is all over
# place and have a singular type of currency throughout our analysis. 

kickstarter['goal'] = kickstarter['goal'] * kickstarter['static_usd_rate']
kickstarter['goal'].head


# In[ ]:


# We want to see the relationship with a kickestarter's initial goal amount versus the amount they did raise. We will
# make a plot for each state. 

# Chart WITHOUT zooming on the scatter plot

for state in kickstarter['state'].unique():
    fig, ax = plt.subplots()
    subset = kickstarter[kickstarter['state'] == state]
    ax.scatter(subset['goal'], subset['usd_pledged'])
    ax.set_title(f'Kickestarter State: {state}')
    ax.set_xlabel('Kickstarter USD Goal Amount')
    ax.set_ylabel('USD Pledge')
plt.show()


# In[ ]:


# We want to see the relationship with a kickestarter's initial goal amount versus the amount they did raise. We will
# make a plot for each state. 

# Chart WITH zooming on the scatter plot

for state in kickstarter['state'].unique():
    fig, ax = plt.subplots()
    subset = kickstarter[kickstarter['state'] == state]
    ax.scatter(subset['goal'], subset['usd_pledged'])
    ax.set_title(f'Kickestarter State: {state}')
    ax.set_xlabel('Kickstarter USD Goal Amount')
    ax.set_ylabel('USD Pledge')
    
    lims = [0, max(subset[['goal','usd_pledged']].max())]
    ax.plot(lims, lims, 'r--', alpha=0.7)
    
    if(state == 'failed'):
        ax.set_ylim(0, 300000)
        ax.set_xlim(0, 300000)
        
    if(state == 'successful'):
        ax.set_xlim(0, 1000000)
        ax.set_ylim(0, 1000000)
        
    if(state == 'live'):
        ax.set_xlim(0, 200000)
        ax.set_ylim(0, 200000)
        
    if(state == 'canceled'):
        ax.set_xlim(0, 1000000)
        ax.set_ylim(0, 600000)
    if(state == 'suspended'):
        ax.set_xlim(0, 500000)
        ax.set_ylim(0, 600000)
        
plt.show()

'''
    So I basically zoomed on the kickestarter's failed and successful state. I set the same scaling on the axises to 
    better compare the goal amount to raise and the amount they were able to raise. I specifically zoomed on where there
    is large portion of data resides and ignore the outliers. 
    
    From the two graphs it is very apparent that if majority of the kickstarters that failed were not able to raise enough
    money. Inversly, kickstarters that were able to raise enough money and many pass their goal amount pointed to their success.
    From these comparison, it is important to keep both goal and usd_pledge amount as a feature to determine the if a kickstarter
    fails or not. 
    
    For the live state, there seems to be no apparent pattern.
    
    For canceled and suspended states, we can see similar behavior like that failed plot.
    
    The line seen on the graph indicates that if a kickstarter falls on the line, they raised their goal amount almost exactly.
'''


# In[ ]:


# Now I want to observe anything between the number of days when a kickstarter was created to when it 
# was launched and the state of the kickstarter.

for state in kickstarter['state'].unique():
    subset = kickstarter[kickstarter['state'] == state]
    
    counts = subset['create_to_launch_days'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(counts.index, counts.values, width=1.0, edgecolor='black')
    
    ax.set_title(f'# of Days from Created to Launch — State: {state}')
    ax.set_xlabel('Duration (days)')
    
    ax.set_xlim(0, 75)
    
    plt.tight_layout()
    plt.show()
    
''' 
    They all seem to the same distribution regardless of the state of the kickstarter. There are initially a high
    count of kickstarters with like 0 to days from create to launch days then exponential decay of the number of kickstarters
    and the days from created to launch. 
    
    I zoomed up to 75 days because that is where majority of the kickstarters concentrated. 
'''


# In[ ]:


for state in kickstarter['state'].unique():
    subset = kickstarter[kickstarter['state'] == state]
    
    counts = subset['launch_to_deadline_days'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(counts.index, counts.values, width=1.0, edgecolor='black')
    
    ax.set_title(f'# of Days from Launch to Deadline — State: {state}')
    ax.set_xlabel('Duration (days)')
    
    ax.set_xlim(0, counts.index.max()+1)
    
    plt.tight_layout()
    plt.show()


''' 
    Observing the graphs for all of the kickstarter states, it seems that there are just a large concentration of 
    kickstarters that fall under 30 days. 
    
    Now I want to see the histogram of just the days from launch to deadline of the entire sample space of the 
    dataset, since for each state, they basically all have 30 days from launch to deadline. 
'''


# In[ ]:


# A histogram the displays the entire dataset of the input days from launch_to_deadline_days 
plt.figure(figsize=(10, 6))
kickstarter['launch_to_deadline_days'].value_counts().plot(kind='bar')
plt.title('Number of Days from Launch to Deadline')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.xlim(0,50)
plt.show()

''' 
    From this, we can see that majority of the kickstarters will have a launch to deadline day of 
    30 days. This feature may play very little in our model.
'''


# In[ ]:


for state in kickstarter['state'].unique():
    subset = kickstarter[kickstarter['state'] == state]
    
    counts = subset['launch_to_state_change_days'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(counts.index, counts.values, width=1.0, edgecolor='black')
    
    ax.set_title(f'# of Days from Launch to Deadline — State: {state}')
    ax.set_xlabel('Duration (days)')
    
    ax.set_xlim(0, counts.index.max()+1)
    
    plt.tight_layout()
    plt.show()
    
'''
    This could be a good feature to include since for the suspended or canceled states, we tend to see days
    before day 30. Since the SuccessfulBool categorizes theses states to be failed, then it might help 
    to classify kickstarters. If we only take a look at the failed and successful states, we primarily see 
    the days fall under day 30, which might be a common theme among all kickstarters as seen in launch to deadlines.
    
    So we are assuming that kickstarters who hit 30 days for a deadline after they launch it could
    mean that they either failed or succeeded. If they fall any number below than that we are sure that 
    those kickstarters failed. 
'''


# In[ ]:


counts = kickstarter.groupby(['state','staff_pick']).size().unstack(fill_value=0)
for country in counts.index:
    freqs = counts.loc[country]
    plt.figure()
    plt.bar(freqs.index, freqs.values)
    plt.title(f'Kickstarter State Counts in {country}')
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
'''
    This may play a little in classifying which kickstarter is successful or not since still majority of the kickstarteres
    in each state were not staff picks
'''


# In[ ]:


counts = kickstarter.groupby(['state','spotlight']).size().unstack(fill_value=0)
for country in counts.index:
    freqs = counts.loc[country]
    plt.figure()
    plt.bar(freqs.index, freqs.values)
    plt.title(f'Kickstarter State Counts in {country}')
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
'''
    We should definately include this. If a kickstarter was in the spotlight, then this would mean 
    the kickstarter would succeed. Looking at the other states, they were all not on the spotlight.
    
    This is a really strong feature to use. 
'''


# In[ ]:


# I am curious about the length of a kickstarter's name and they're successful

for state in kickstarter['state'].unique():
    subset = kickstarter[kickstarter['state'] == state]
    
    counts = subset['name_len_clean'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(counts.index, counts.values, width=1.0, edgecolor='black')
    
    ax.set_title(f'Length of Kickstarter Name: {state}')
    ax.set_xlabel('Length of Name')
    
    ax.set_xlim(0, 15)
    
    plt.tight_layout()
    plt.show()
    
'''
    From looking at the distributions for each state, they all share a similar Guassian Distribution. However, specifically for the failed states, 
    we can see that the distribution is relatively evenly distributed for each column below a length of 7. So it is similar to a right-skewed distribution.
    Regardless, I think this column will play a little impact on the successful rate of a kickstarter.
'''


# In[ ]:


# I am curious about the length of a kickstarter's blurb and they're successful

for state in kickstarter['state'].unique():
    subset = kickstarter[kickstarter['state'] == state]
    
    counts = subset['blurb_len_clean'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(counts.index, counts.values, width=1.0, edgecolor='black')
    
    ax.set_title(f'Length of Kickstarter Name: {state}')
    ax.set_xlabel('Length of Name')
    
    ax.set_xlim(0, 15)
    
    plt.tight_layout()
    plt.show()


# In[ ]:


counts = kickstarter.groupby(['country', 'TOPCOUNTRY']).size().unstack(fill_value=0)

country_sorted = counts.index.tolist()
states = counts.columns.tolist()
x = np.arange(len(country_sorted))
width = 0.8 / len(states)

# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))
for i, cat in enumerate(states):
    ax.bar(x + i*width, counts[cat], width=width, label=cat)

ax.set_xlabel('Country')
ax.set_ylabel('Number of Projects')
ax.set_title('Kickstarter Projects by Country and TOPCOUNTRY')
ax.set_xticks(x + width*(len(states)-1)/2)
ax.set_xticklabels(years_sorted, rotation=45)
ax.legend(title='Category')
plt.tight_layout()
plt.show()

# So TOPCOUNTRY includes United States, Great Britain, Mexico, Hong Kong, New Zealand


# In[ ]:


counts = kickstarter.groupby(['state','TOPCOUNTRY']).size().unstack(fill_value=0)
for binary in counts.index:
    freqs = counts.loc[binary]
    plt.figure()
    plt.bar(freqs.index, freqs.values)
    plt.title(f'Kickstarter State Counts in {binary}')
    plt.xlabel('TOPCOUNTRY')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
# There is no noticeable difference between each state, other than the fact that in the successful state, there is a smaller ratio between
# if the top country is yes or no. Since we have the country, this feature is redundant. BUT, since there is significantly less kickstarters that 
# are not a top country and succeeded, can help improve our bias model. This dataset is very biased. 


# In[ ]:


counts = kickstarter.groupby(['state','LaunchedTuesday']).size().unstack(fill_value=0)
for binary in counts.index:
    freqs = counts.loc[binary]
    plt.figure()
    plt.bar(freqs.index, freqs.values)
    plt.title(f'Kickstarter State Counts in {binary}')
    plt.xlabel('LaunchedTuesday')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Again, no pattern, useless feature to drop.


# In[ ]:


counts = kickstarter.groupby(['state','DeadlineWeekend']).size().unstack(fill_value=0)
for binary in counts.index:
    freqs = counts.loc[binary]
    plt.figure()
    plt.bar(freqs.index, freqs.values)
    plt.title(f'Kickstarter State Counts in {binary}')
    plt.xlabel('DeadlineWeekend')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Again, no pattern, useless feature to drop.


# In[ ]:


'''
    From our understanding of the data, we can clear cut which inputs to use for our models.
    
    Note, the relationship between goal and pledge is a STRONG indicator if a kickstarter would fail or not. 
    Another one is if a kickstarter is in the spotlight. This is a VERY VERY strong indicator if a kickstarter
    would fail or not. We could essentially rely on these two.
    
    The unfortunate thing is that pledge is the amount they raised it is clear that if a kickstarter raised more than
    their goal amount, then the kickstarter will succeed. We want to look at the kickstarter with their initial information,
    so we cannot look at their pledge amount, and number of supports (backers_count)
'''

selected_inputs = ['goal', 'country', 'staff_pick', 'backers_count', 'usd_pledged', 'category', 'spotlight', 
                   'created_at_yr', 'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days']

output = ['SuccessfulBool']

selected_columns = ['goal', 'state', 'country', 'staff_pick', 'backers_count', 'usd_pledged', 'category', 'spotlight', 
                   'created_at_yr', 'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days', 'SuccessfulBool']


# In[ ]:


kickstarter = kickstarter[selected_columns]


# In[ ]:


to_encode = ['category', 'country']

encoder = LabelEncoder()
for col in to_encode:
    kickstarter[col] = encoder.fit_transform(kickstarter[col])


# In[ ]:


kickstarter.head


# In[ ]:


kickstarter = kickstarter[kickstarter['state'] != 'live']

# Seeing all of the count for each kickstarter state
plt.figure(figsize=(10, 6))
kickstarter['state'].value_counts().plot(kind='bar')
plt.title('Kickstarter State Count')
plt.ylabel('Count')
plt.ylim(0, 12000)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


print(len(kickstarter))

