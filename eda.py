import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
filepath = "E:/vs/Kickstarter_Project/dataset/kickstarter_data_full.csv"
kickstarter = pd.read_csv(filepath)
kickstarter.head()
kickstarter.shape
kickstarter.columns
toDrop = ["Unnamed: 0", "id", "photo", "name", "blurb", "slug", "creator", "location", 
        "profile", "urls", "source_url", "friends", "is_starred", "is_backing", "permissions", 
        "name_len", "name_len_clean", "blurb_len", "blurb_len_clean", "deadline", 
        "state_changed_at", "created_at", "launched_at", "create_to_launch", "launch_to_deadline", 
        "launch_to_state_change", "currency_symbol", "deadline_weekday", "disable_communication", 
        "static_usd_rate"]

toEncode = ["state", "country", "currency", 
            "currency_trailing_code", "staff_pick", "category", "spotlight", 
            "state_changed_at_weekday", "created_at_weekday", "launched_at_weekday"]

'''
    Catergorical columns like `id`, `photo`, and `name` are dropped because they present unique stings 
    that have no correlation with other other rows meaning that if we encoded them, we would get 20632 
    unique ids. 
    
    There are also some that seem completly useless like `name_len`. 
    
    Columns with dates and other useful information like `deadline`, `created_at`, `launched_at` could 
    be filtered into a new integer column for use. Thankfully the dataset provides these for example 
    splitting the string column `deadline` into an int `deadline_month`, `deadline_year`, and `deadline_day`.
    
    Other columns like `disable_communication` will be removed as well as they dont seem to hold much correlation
    to other features and their a majority of its data is false and assuming the minority are outliers. 
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
plt.figure()
heapmap = clean_kickstarter.corr(method='pearson')
sns.heatmap(heapmap.round(2), square=True, annot=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.rcParams['figure.figsize'] = [25, 30]
plt.show()

clean_kickstarter.corr()["SuccessfulBool"].sort_values()

scatter_matrix(clean_kickstarter)
plt.show()

'''
    Testing out some transforms
'''

x = clean_kickstarter.drop(["SuccessfulBool"], axis=1)
y = pd.DataFrame(clean_kickstarter["SuccessfulBool"])

standardizer = StandardScaler().fit(x)
normalizer = MinMaxScaler().fit(x) 
# used minmax cause normalizer wasnt showing on graphs and log would turn some values to 0, but represent it as NaN

x_nrm = pd.DataFrame(normalizer.transform(x), columns=x.columns)
df_nrm = pd.concat([x_nrm*3, y], axis=1) # I scaled this so I could see it better on the graphs

x_std = pd.DataFrame(standardizer.transform(x), columns=x.columns)
df_std = pd.concat([x_std, y], axis=1)

df_nrm.info()

columns_to_plot = x.columns

# Plot original df, standardized, and normalized data on histograms
fig, axes = plt.subplots(11, 4, figsize=(20, 5 * 4))
axes = axes.flatten() 
for i, col in enumerate(columns_to_plot):
    ax = axes[i]
    ax.hist(clean_kickstarter[col], bins=30, alpha=0.4, label='Original')
    ax.hist(df_std[col], bins=30, alpha=0.4, label='Standardized')
    ax.hist(df_nrm[col], bins=30, alpha=0.4, label='Normalized')
    ax.set_title(col)
    ax.legend(fontsize='small')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Plot log and sqrt df
df_srqt_log = clean_kickstarter
for col in x.columns:
    df_srqt_log[col] = np.log2(np.sqrt(df_srqt_log[col]))
df_srqt_log.replace(-np.inf, 0, inplace=True)
df_srqt_log.hist()

# I like the distribution for df_srqt_log so I will be going with this for now and can invesitgate more translations later
# We will also export these variables to use in our training and testing
df = df_srqt_log
X = df.drop(["SuccessfulBool"], axis=1)
Y = pd.DataFrame(df["SuccessfulBool"])