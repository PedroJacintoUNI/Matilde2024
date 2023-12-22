import lifelines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import load_data

from lifelines import KaplanMeierFitter as KM
from lifelines.statistics import logrank_test


from public_tests import *

data = load_data()
