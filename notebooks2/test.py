import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import re
#import random
#from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
#from string import ascii_letters
import seaborn as sns
import h5py as h5


trial_res = pd.read_csv('out.csv')

plt.figure()
sns.set(font_scale = 3)
sns.pairplot(trial_res, x_vars=['n_estimators', 'colsample_bytree', 'max_depth', 'min_child_weight', 'reg_alpha', 'reg_lambda' ], y_vars=['train_accuracy','val_accuracy','delta_accuracy'], kind='reg',  height=5, hue = 'Selection')

plt.show()



