
# THIS PYTHON SCRIPT TAKES IN ARGUMENT THE PATH TO A PNG IMAGE 
# AND PRINT THE SKILL ASSESSED 


################## IMPORT #############################
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import pickle
from sklearn.externals import joblib
import sys

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

####################### USEFUL FUNCTION TO READ IMAGE ###################

def image_to_list(image):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()


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
        if ( liste_features[i] in X_list):
            X_array[0,i]=1
    
    return X_array


def label_number(liste_label, Y):
    Y_label=Y.copy()
    
    for i in range(len(Y)):
        Y_label[i,0]=liste_label.index(Y[i,0])
        
    return Y_label

    
############### WE IMPORT THE LIST OF FEATURES #########################



######################################################################
##============================= MAIN FUNCTION ======================
if __name__ == '__main__':

    # import the list of features
    with open('liste_features.pkl', 'rb') as pickle_load:
        liste_features = pickle.load(pickle_load)

    loaded_model = joblib.load('finalized_model.sav')

    filename=sys.argv[1]
    with io.open(filename, 'rb') as image_file:
            content = image_file.read()
    image = types.Image(content=content)



    # #convert the image into an array using bag of words
    X_exp=list_to_array(image_to_list(image), liste_features)
    y_predict=np.asarray(loaded_model.predict(X_exp)).reshape(-1,1)
    print('Skill=',y_predict[0][0])











