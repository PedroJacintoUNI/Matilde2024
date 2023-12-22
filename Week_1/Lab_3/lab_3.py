#import pandas 
import pandas as pd

#import a pre-defined function that generates data
from utils import load_data 

#Generate features and labels
X, y = load_data(100)

X.head()

feature_names = X.columns
feature_names

name1 = feature_names[0]
name2 = feature_names[1]

print(f"name1: {name1}")
print(f"name2: {name2}")
# Combine the names of two features into a single string, separated by '_&_' for clarity
combined_names = f"{name1}_&_{name2}"
combined_names

#Add two columns
X[combined_names] = X['Age'] + X['Systolic_BP']
X.head(2)

# Generate a small dataset with two features
df = pd.DataFrame({'v1': [1,1,1,2,2,2,3,3,3],
                   'v2': [100,200,300,100,200,300,100,200,300]
                  })

# add the two features together
df['v1 + v2'] = df['v1'] + df['v2']

# multiply the two features together
df['v1 x v2'] = df['v1'] * df['v2']
df


# Import seaborn in order to use a heatmap plot
import seaborn as sns

#Import matplotlib
import matplotlib.pyplot as plt

# Pivot the data so that v1 + v2 is the value

df_add = df.pivot(index='v1',
                  columns='v2',
                  values='v1 + v2'
                 )
print("v1 + v2\n")
print(df_add)
plt.figure(1)
ax = sns.heatmap(df_add)


#Pivot the data so that v1 * v2 is the value
df_mult = df.pivot(index='v1',
                  columns='v2',
                  values='v1 x v2'
                 )
print('v1 x v2')
print(df_mult)
plt.figure(2)
m = sns.heatmap(df_mult)
plt.show()


