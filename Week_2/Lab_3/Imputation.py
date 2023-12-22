import pandas as pd
import numpy as np

df = pd.DataFrame({"feature_1": [0,1,2,3,4,5,6,7,8,9,10],
                   "feature_2": [0,np.NaN,20,30,40,50,60,70,80,np.NaN,100],
                  })
df


#MEAN IMPUTATION

from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
mean_imputer

mean_imputer.fit(df)


nparray_imputed_mean = mean_imputer.transform(df)
print(nparray_imputed_mean)

#REGRESSION IMPUTATION

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

reg_imputer = IterativeImputer()
reg_imputer

reg_imputer.fit(df)

nparray_imputed_reg = reg_imputer.transform(df)
print(nparray_imputed_reg)
