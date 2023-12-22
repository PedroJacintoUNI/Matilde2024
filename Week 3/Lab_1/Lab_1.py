import numpy as np
import pandas as pd


df = pd.DataFrame({'Time': [10,8,60,20,12,30,15],
                   'Event': [1,0,1,1,0,1,0]
                  })
df

df['Event'] == 0

sum(df['Event'] == 0)

t = 25
df['Time'] > t

sum(df['Time'] > t)

t = 25
(df['Time'] > t) | (df['Event'] == 0)

sum( (df['Time'] > t) | (df['Event'] == 0) )

t = 25
(df['Event'] == 1) | (df['Time'] > t)


sum( (df['Event'] == 1) | (df['Time'] > t) )