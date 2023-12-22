import numpy as np
import pandas as pd

df = pd.DataFrame({"feature_1": [0.1,np.NaN,np.NaN,0.4],
                   "feature_2": [1.1,2.2,np.NaN,np.NaN]
                  })
df

#Check if any value is missing
df.isnull()

#Chack if any values in a row are true
df_booleans = pd.DataFrame({"col_1": [True,True,False],
                            "col_2": [True,False,False]
                           })
df_booleans

#If we use pandas.DataFrame.any(), it checks if at least one value in a column is `True`, and if so, returns `True`.
#If all rows are `False`, then it returns `False` for that column

df_booleans.any()

df_booleans.any(axis=0)

df_booleans.any(axis=1)


series_booleans = pd.Series([True,True,False])
series_booleans

sum(series_booleans)

#Apply a mask

import pandas as pd
df = pd.DataFrame({"feature_1": [0,1,2,3,4]})
df

mask = df["feature_1"] >= 3
mask

df[mask]

#Combining comparision operatores
df["feature_1"] >=2

df["feature_1" ] <=3

# This will compare the series, one row at a time
(df["feature_1"] >=2) & (df["feature_1" ] <=3)