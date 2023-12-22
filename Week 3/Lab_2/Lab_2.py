import numpy as np
import pandas as pd

df = pd.DataFrame({'Time': [3,3,2,2],
                   'Event': [0,1,0,1]
                  })
df


t_i = 2
df['Time'] >= t_i

t_i = 2
(df['Event'] == 1) & (df['Time'] == t_i)