
#This Python programm create a csv file that can be used for training 



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
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
####################### 




 #################  Detects document features in an image ################
#This function will read the words present in an image and store the words in
# a list.def detect_document(path):
def image_to_list(path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    word_list=[]
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    mot=''
                    for symbol in word.symbols:
                        mot=mot+symbol.text
                    word_list.append(mot)

            
            
    return word_list

 ####### Function that transform a list of words into a vector of features ######
def list_to_array(X_list,liste_features):
    
    X_array=np.zeros((1,len(liste_features)))

    for i in range(len(liste_features)):
        #block_word=liste_features[i]
        #for j in range(len(block_word)):
        if ( liste_features[i] in X_list):
            X_array[0,i]=1
    
    return X_array

    

######### this function return a list of all the words in the folder with all exercises ######
def transform_pictures_to_array(n):
    image_list = []
    compt=1
    for filename in glob.glob('mix_images/*.png'):

        image_list=image_list+image_to_list(filename)
        compt=compt+1
        if(compt%200==0):
            print(compt/1435*100, " %") 
    return image_list

################   this function takes out the numbers of a list and repetitions ##########
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def no_numbers(liste):
    new_liste=liste.copy()
    liste_int=[]
    for i in range(len(liste)):
        if(RepresentsInt(new_liste[i])==True):
            liste_int.append(new_liste[i])
    final_list= list(set(new_liste).difference(set(liste_int)))
    return final_list

def no_repetition(liste):
    return list(set(liste))


############# Construct the vectorize data set ##################
import re
def training_data(liste_features):
    l=len(liste_features)
    y_train=[]
    X_train=np.zeros((1,len(liste_features)))
    compt=1
    for filename in glob.glob('mix_images/*.png'):
        label=filename.rsplit('_', 1)[0]
        y_train.append(label.rsplit('/', 2)[1])
        #y_train.append(label)

        X_list=image_to_list(filename)
        X_array=list_to_array(X_list,liste_features)
        X_train=np.concatenate((X_train, X_array), axis=0)
        # savoir ou j'en suis
        compt=compt+1
        if(compt%200==0):
            print(compt/1435*100,'%')

    y_train=np.asarray (y_train)
    y_train.reshape(-1,1)

    return X_train[1:,:] , y_train



########## create label vector ####################################

def labels(n):
    # list with all the filename in images_test
    file_list = []
    y=[]
    for filename in glob.glob('mix_images/*.png'):
        file_list.append(filename)
     

    for fname in file_list:
        label =  fname.rsplit('_', 2)[0]
        label=label.rsplit('/', 2)[1]
        y.append(label)
    y=np.asarray (y)
    y.reshape(-1,1)
    return y


#####################  Saving the raw data   ##########################################
print('-------- Create liste_features and save into text file ----------')    
liste_features=no_repetition(no_numbers(transform_pictures_to_array(3)))
print('size of liste_feature:',len(liste_features))

print('-------- Create a training array ----------')
X_train, y_train=training_data(liste_features)
print(X_train.shape, y_train.shape)

print('-------- saving training into csv file ----------')
df_train= pd.DataFrame(data=X_train, index=None ,columns=liste_features)
df_label=pd.DataFrame(data=y_train, index=None, columns=['type_skill'])

df_train.to_csv('Exo_train_mix.csv')
df_label.to_csv('label_exo_train_mix.csv')


print(' ========Size of the dataset: ', X_train.shape,' elements')
print(y_train.shape)

############## PREPROCESSING #########################
def label_number(liste_label, Y):
    Y_label=Y.copy()
    
    for i in range(len(Y)):
        Y_label[i,0]=liste_label.index(Y[i,0])
        
    return Y_label

# This function removes the useless features of liste_features
def reduction_list(liste_id_remove, liste_features):
    
    reduced_list=[i for j, i in enumerate(liste_features) if j not in liste_id_remove]
    return reduced_list

def feature_to_remove(coef):
    list_id=[]
    for i in range(coef.shape[1]):
        if (np.linalg.norm(coef[:,i])<1e-5):
            list_id.append(i)
    return list_id
    
    
#import the data
print("============= Import Data =================")
df_train=pd.read_csv('Exo_train_mix.csv')
df_train=df_train.drop(df_train.columns[0], axis=1)
df_label=pd.read_csv('label_exo_train_mix.csv')
df_label=df_label.drop(df_label.columns[0], axis=1)






print('============ GET RID OF WORDS THAT APPEAR LESS THAN 3 TIMES ============')
# get ride of words that appear less than 3 times
words=df_train.columns.values
drop=[]
drop_id=[]
for i in range(len(words)):
    if (sum(df_train[str(words[i])])<3):
        drop.append(words[i])
        liste_features.remove(words[i])
        drop_id.append(i)
    
print( " new size liste_features:", len(liste_features))

df_train=df_train.drop(drop, axis=1) 

X= df_train.as_matrix()
Y=df_label.as_matrix()
liste_label=  list(set(Y.reshape(1,len(Y))[0]))
Y_label=label_number(liste_label, Y)

# ################# Keep the best features #######################
print('========== KEEPING THE BEST FEATURES USING L1 NORM WITH LINEAR SVC ==========')

lsvc=LinearSVC(C=0.3, penalty="l1", dual=False).fit(X, Y.ravel())
model = SelectFromModel(lsvc, prefit=True)
print('Shape of X : ', X.shape)
X = model.transform(X)
print('Shape of X after feature reduction: ', X.shape)

liste_id_remove=feature_to_remove(lsvc.coef_)
features_removed=[]
for id in liste_id_remove:
    features_removed.append(liste_features[id])

print('\n========== REMOVED FEATURES ============== \n\n\ ',features_removed)

liste_features=reduction_list(liste_id_remove, liste_features)





print('\n========== FINAL SIZE OF X ===============')
print('size of X_train:', X.shape)
print('size of Y_train:', Y.shape)

####################### SAVE LISTE_FEATURES ##############
print(liste_features)
with open('liste_features.pkl', 'wb') as pickle_file:
    pickle.dump(liste_features, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)



df_train= pd.DataFrame(data=X, index=None ,columns=liste_features)
df_label=pd.DataFrame(data=Y, index=None, columns=['type_skill'])

df_train.to_csv('Exo_train_mix_cleaned.csv')
df_label.to_csv('label_exo_train_mix_cleaned.csv')















