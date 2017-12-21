
# This function traines a machine learning model and save it ####


################## IMPORT #############################
import numpy as np
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import io
import os
import simplejson
from PIL import Image
import glob 

####################### 



    
    
####### IMPORT DATA ###################################
print("============= Import Data =================")
df_train=pd.read_csv('Exo_train_mix_cleaned.csv')
df_train=df_train.drop(df_train.columns[0], axis=1)
df_label=pd.read_csv('label_exo_train_mix_cleaned.csv')
df_label=df_label.drop(df_label.columns[0], axis=1)


X_train= df_train.as_matrix()
Y_train=df_label.as_matrix()

print('======shape====')
print(X_train.shape, Y_train.shape)

#### SEARCH FOR BEST PARAMETER (I restricted myself to logistic regression) #############
log =LogisticRegression()
parameters = {'C':[ 20*k for k in range(1,10)]}
log = GridSearchCV(log, parameters)
log.fit(X_train, Y_train.ravel())

C=log.best_params_['C']
print('best parameters:',C)
############## TRAIN ANS SAVE MODEL ############################
log2 =LogisticRegression(C=C)
log2.fit(X_train, Y_train.ravel())
filename = 'finalized_model.sav'
joblib.dump(log2, filename)


















