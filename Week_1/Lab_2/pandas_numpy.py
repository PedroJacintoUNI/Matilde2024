# Import packages
import numpy as np
import pandas as pd

# Import a predefined function that will generate data
from Lab_3.utils import load_data

# generate the features 'X' and labels 'y'
X, y = load_data(100)

# View the first few rows and column names of the features data frame
X.head()

#view the labels
y.head()

# Call the .mean function of the data frame without choosing an axis
print(f"Pandas: X.mean():\n{X.mean()}")
print()
# Call the .mean function of the data frame, choosing axis=0
print(f"Pandas: X.mean(axis=0)\n{X.mean(axis=0)}")

#For pandas DataFrames:
#- By default, pandas treats each column separately.  
#- You can also explicitly instruct the function to calculate the mean for each column by setting axis=0.
#- In both cases, you get the same result.

# Store the data frame data into a numpy array
X_np = np.array(X)

# view the first 2 rows of the numpy array
print(f"First 2 rows of the numpy array:\n{X_np[0:2,:]}")
print()

# Call the .mean function of the numpy array without choosing an axis
print(f"Numpy.ndarray.mean: X_np.mean:\n{X_np.mean()}")
print()
# Call the .mean function of the numpy array, choosing axis=0
print(f"Numpy.ndarray.mean: X_np.mean(axis=0):\n{X_np.mean(axis=0)}")

#Notice how the default behavior of numpy.ndarray.mean differs.
#- By default, the mean is calculated for all values in the rows and columns.  You get a single mean for the entire 2D array.
#- To explicitly calculate the mean for each column separately, you can set axis=0.


