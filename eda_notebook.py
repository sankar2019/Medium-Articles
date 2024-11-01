#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import matplotlib as mplot
import seaborn as sns
import matplotlib.pyplot as plt


# In[371]:


netflix = pd.read_csv('netflix_dataset/netflix_titles_2021.csv')


# In[15]:


# First 5 rows
print(netflix.head())

# Last 5 rows
print(netflix.tail())

# Random sample of data
print(netflix.sample(5))


# In[23]:


# Display first 5 records of the Netflix dataset
netflix.head(5)


# In[25]:


# Display last 5 records of the Netflix dataset
netflix.tail(5)


# In[21]:


# Display random 5 records from the Netflix dataset
netflix.sample(5)


# In[11]:


netflix.describe(include='all')


# In[27]:


# Basic information about columns and data types
print(netflix.info())

# Summary statistics
print(netflix.describe(include='all'))


# In[29]:


# Basic information about columns and data types
netflix.info()


# In[31]:


# Summary statistics
netflix.describe(include='all')


# In[33]:


netflix.describe()


# In[37]:


# Total missing values per column
netflix.isnull().sum()


# In[51]:


# Visualize missing values
sns.heatmap(netflix.isnull(), cbar=False, cmap="viridis")
plt.show()


# In[53]:


# Calculate the number of missing values in each column
missing_values = netflix.isnull().sum()

# Plot the missing values using a bar chart
missing_values[missing_values > 0].plot(kind='bar', color='salmon')
# Let's give a title to the plot and to the X and Y labels
plt.title("Missing Values per Column")
plt.xlabel("Columns")
plt.ylabel("Number of Missing Values")
plt.show()


# In[55]:


# Calculate the percentage of missing values for each column
missing_percentage = (netflix.isnull().mean() * 100)

# Plot the missing percentage using a bar chart
missing_percentage[missing_percentage > 0].plot(kind='bar', color='skyblue')
# Let's give a title to the plot and to the X and Y labels
plt.title("Percentage of Missing Values per Column")
plt.xlabel("Columns")
plt.ylabel("Percentage of Missing Values (%)")
plt.show()


# In[85]:


# Fill missing values in country column with mode
netflix['country'].fillna(netflix['country'].mode()[0], inplace=True)


# In[369]:


netflix['cast'].fillna(netflix['cast'].mode()[0], inplace=True)
netflix['date_added'].fillna(netflix['date_added'].mode()[0], inplace=True)
netflix['duration'].fillna(netflix['duration'].mode()[0], inplace=True)
netflix['rating'].fillna(netflix['rating'].mode()[0], inplace=True)


# In[129]:


# Calculate the percentage of missing values for each column
missing_percentage = (netflix.isnull().mean() * 100)

# Plot the missing percentage using a bar chart
missing_percentage[missing_percentage > 0].plot(kind='bar', color='skyblue')
# Let's give a title to the plot and to the X and Y labels
plt.title("Percentage of Missing Values per Column")
plt.xlabel("Columns")
plt.ylabel("Percentage of Missing Values (%)")
plt.show()


# In[249]:





# In[373]:


#import random forest classifier package for prediction and label encoder package for encoding of categorical columns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# lets sample the dataset to reduce the number of records
netflix = netflix.sample(1000)

# Dictionary to store encoders
encoders = {}

# Iterate through each categorical column and encode them into numerical so that we can make predictions on them
for col in netflix.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    netflix[col + '_encoded'] = le.fit_transform(netflix[col])
    encoders[col] = le  # Store the encoder for inverse transformation later

# Define the substring and prediction column name for filtering
substring = '_encoded'
additional_column = 'director'

# Use filter for substring and add the extra column
netflix_subset = netflix.filter(like=substring)

# Check if the additional column exists before adding it
if additional_column in netflix.columns and additional_column not in netflix_subset.columns:
    netflix_subset[additional_column] = netflix[additional_column]

# Separate data with known and unknown values in the categorical column
known_data = netflix_subset[netflix_subset['director'].notna()]
unknown_data = netflix_subset[netflix_subset['director'].isna()]

# Define features and target for the known data
X_known = known_data.loc[:, known_data.columns != 'director']
y_known = known_data['director_encoded']

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_known, y_known)

# Predict missing categorical values in unknown data
X_unknown = unknown_data.loc[:, known_data.columns != 'director']
predictions = rf.predict(X_unknown)

# Fill the missing values in the original DataFrame
netflix.loc[netflix['director'].isna(), 'director_encoded'] = predictions

# perform inverse transform of encoded columns to get the actual column names
for col, le in encoders.items():
    netflix[col + '_decoded'] = le.inverse_transform(netflix[col + '_encoded'])


# In[374]:


netflix[['director','director_encoded','director_decoded']]


# In[269]:


netflix['rating'].dropna()


# In[271]:


netflix['director']


# In[281]:


# Number of duplicates to create
n_duplicates = 5

# Sample random rows with replacement and append to the DataFrame
duplicates = netflix.sample(n=n_duplicates, replace=True)
netflix = pd.concat([netflix, duplicates], ignore_index=True)


# In[283]:


netflix[netflix.duplicated()]


# In[285]:


# Drop duplicates
netflix.drop_duplicates(inplace=True)


# In[377]:


# Box plot for visualizing outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=netflix['rating_encoded'])
plt.title("Box Plot to Identify Outliers")
plt.show()


# In[299]:


from scipy.stats import chi2_contingency

# Create a contingency table for two categorical variables
contingency_table = pd.crosstab(netflix['rating'], netflix['duration'])

# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Compare observed vs expected values to spot outliers
observed = contingency_table.values
outliers = np.where((observed - expected) > 2)  # Threshold depends on context

print("Outliers based on Chi-Square test:", outliers)


# In[305]:


netflix_subset.drop(columns={'director'}, inplace=True)


# In[379]:


# IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # First quartile
    Q3 = df[column].quantile(0.75)  # Third quartile
    IQR = Q3 - Q1  # Interquartile range
    
    # Calculate the bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers
    
column = 'rating_encoded'
outliers = detect_outliers_iqr(netflix, column)
if len(outliers[column])>0:
    print(f"Outliers in {column} based on IQR:\n", outliers[column])
    print(netflix.loc[list(outliers[[column]].index)][[column,column.removesuffix('_encoded')+'_decoded']])


# In[381]:


outliers['rating_decoded'].unique()


# In[321]:


column = 'rating_encoded'


# In[325]:


column.removesuffix('_encoded')


# In[351]:


netflix.loc[list(outliers[[column]].index)][[column,column.removesuffix('_encoded')+'_decoded']]


# In[389]:


# Define a threshold (e.g., remove values more than 2 standard deviations from the mean)
threshold = 2
mean, std_dev = netflix['rating_encoded'].mean(), netflix['rating_encoded'].std()
netflix = netflix[(netflix['rating_encoded'] < mean + threshold * std_dev) & (netflix['rating_encoded'] > mean - threshold * std_dev)]


# In[391]:


# Box plot for visualizing outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=netflix['rating_encoded'])
plt.title("Box Plot to Identify Outliers")
plt.show()


# In[393]:


from scipy.stats.mstats import winsorize

# Winsorize data by capping at the 5th and 95th percentiles
netflix['duration_encoded'] = winsorize(netflix['duration_encoded'], limits=[0.05, 0.05])


# In[395]:


# Box plot for visualizing outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=netflix['duration_encoded'])
plt.title("Box Plot to Identify Outliers - duration")
plt.show()


# In[399]:


# Use filter for substring and add the extra column
netflix_subset = netflix.filter(like='_encoded')


# In[403]:


netflix_subset.drop(columns={'show_id_encoded'}, inplace=True)


# In[405]:


# Remove substring "_old" from column names
netflix_subset.columns = netflix_subset.columns.str.replace('_encoded', '', regex=False)


# In[407]:


netflix_subset.columns


# In[409]:


# Step 1: Calculate the Correlation Matrix
correlation_matrix = netflix_subset.corr()

# Step 2: Display the Correlation Matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Step 3: Visualize the Correlation Matrix using a Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap")
plt.show()


# In[ ]:




