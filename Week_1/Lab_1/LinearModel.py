#Import the module 'LinearRegression' from sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Create an objevt of type LinearRegression
model = LinearRegression()
model

# Import the load_data function from the utils module
from Lab_2.utils import load_data

# Generate features and labels using the imported function
X, y = load_data(100)

# View the features
X.head()

# Plot a histogram of the Age feature
#Different way of doing a histogram using the plt.
plt.figure(1)
plt.hist(X['Age'])


# Plot a histogram of the systolic blood pressure feature
plt.figure(2)
#plt.hist(X['Systolic_BP'])
X['Systolic_BP'].hist()
plt.show() #in the last plot

# Plot a histogram of the diastolic blood pressure feature
X['Diastolic_BP'].hist()

# Plot a histogram of the cholesterol feature
X['Cholesterol'].hist()

# View a few values of the labels
y.head()

# Plot a histogram of the labels
y.hist()

# Fit the linear regression model
model.fit(X, y)
model

# View the coefficients of the model
model.coef_
print(model.coef_)


