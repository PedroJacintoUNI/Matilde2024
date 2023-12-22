import pandas as pd
from sklearn.tree import DecisionTreeClassifier

X = pd.DataFrame({"feature_1":[0,1,2,3]})
y = pd.Series([0,0,1,1])

X 
y
dt = DecisionTreeClassifier()
dt
dt.fit(X,y)

#Set the parameters 
dt = DecisionTreeClassifier(criterion='entropy',
                            max_depth=10,
                            min_samples_split=2
                           )
dt

#we can also set the parameteres using a dictionary 
tree_parameters = {'criterion': 'entropy',
                   'max_depth': 10,
                   'min_samples_split': 2
                  }

dt = DecisionTreeClassifier(**tree_parameters)
dt