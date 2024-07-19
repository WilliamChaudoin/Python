import os
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pylab as plt

# Loading Dataframe
plants = pd.read_csv("iris_data.csv")
print(plants.head())

# Examining Dataframe
print(plants.shape[0])
print(plants.columns.tolist())
print(plants.dtypes)

# Looking At Counts
plants.species.value_counts()

# Setting Up Some Descriptive Statistics
df = plants.describe()
df.loc['range'] = df.loc['max'] - df.loc['min']
fields = ['mean','25%','50%','75%', 'range']
df = df.loc[fields]
df.rename({'50%': 'median'}, inplace=True)
print(df)

# Grouping Materials By Metric
## Mean
plants.groupby('species').mean()

## Median
plants.groupby('species').median()

# Alternative Method
plants.groupby('species').agg(['mean', 'median'])

# Creating A Scatterplot
plot1 = plt.axes()
plot1.scatter(plants.sepal_length, plants.sepal_width)
plot1.set(xlabel='Sepal Length (cm)', ylabel='Sepal Width (cm)', title='Sepal Length vs Width');
plt.show()

# Creating A Histogram
plot2 = plt.axes()
plot2.hist(plants.petal_length, bins=25);
plot2.set(xlabel='Petal Length (cm)', ylabel='Frequency', title='Distribution of Petal Lengths');
plt.show()

# Creating Multiple Boxplots
plants.boxplot(by='species')
plt.show()
