import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image, ImageChops, ImageEnhance

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier



# Configuration

TRAIN_DIR = '../input/my-images/my_images'               # folder with 2 subfolders for authentic and modified images
TEST_DIR = '../input/my-images-test-dataset/test_images' # folder with test images
TEMP_DIR= '/kaggle/temporary_images'
ELA_RATE = 85 # error ELA rate used when resaving an image
DECISION_BOUNDARY = 0.5
IMG_WIDTH = 256
IMG_HEIGHT = 256
N_EPOCHS = [10, 20, 40] 
BATCH_SIZES = [16, 32, 64]
DROPOUT_RATES = [0.2, 0.4, 0.5]
SPLIT_RATE = 0.1 # splitting proportion for training and validation datasets


def applyELA(filepath):
    """ Error Level Analysis (ELA) is taken from http://www.hackerfactor.com/papers/bh-usa-07-krawetz-wp.pdf
    Basically it permits identifying areas within an image that are at different compression levels. 
    With JPEG images the entire picture should be at roughly the same level. 
    If a section of the image is at a significantly different error level, then it likely indicates a digital modification.
   
    Args:
        filepath: path to the image to analyze        

    Returns:
        Image object - ELA picture
    """
    if not os.path.isdir(TEMP_DIR): 
        os.mkdir(TEMP_DIR)
    
    temp_filename = TEMP_DIR+'/temporary_image.jpg'
    original = Image.open(filepath).convert('RGB')
    original.save(temp_filename, 'JPEG', quality=ELA_RATE) #resave image at a known error rate, e.g. 95%
    temporary = Image.open(temp_filename)
    
    diff = ImageChops.difference(original, temporary)  # compute the difference between 2 images
    extrema = diff.getextrema()
    
    # if no change, then pixel has reached its local minima for error at that quality level. 
    # if there is a large amount of change, then the pixels are not at their local minima and are effectively original
   
    max_diff = max([ex[1] for ex in extrema])
    max_diff = 1 if max_diff == 0 else max_diff
    scale = 255.0 / max_diff
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    return diff

def processImages(pathToImages):
    """ Function performs ELA processing 
    Args:
        pathToImages: path to the folder with image files      

    Returns:
        X: array with features
        
    """
    X = []
    for img in os.listdir(pathToImages):
        X.append(np.array(applyELA(pathToImages+'/'+img).resize((IMG_WIDTH, IMG_HEIGHT))).flatten() / 255.0)
    return X    

original_train_n = sum([len(files) for r, d, files in os.walk(TRAIN_DIR+'/real')])
edited_train_n = sum([len(files) for r, d, files in os.walk(TRAIN_DIR+'/edited')])
original_test_n = sum([len(files) for r, d, files in os.walk(TEST_DIR+'/real')])
edited_test_n = sum([len(files) for r, d, files in os.walk(TEST_DIR+'/edited')])
print('Number of training images: {0} (original: {1}, modified: {2}) '.format(original_train_n+edited_train_n, original_train_n, edited_train_n))
print('Number of test images: {0} (original: {1}, modified: {2}) '.format(original_test_n+edited_test_n, original_test_n, edited_test_n))


# Preparing training dataset

X_train_real = processImages(TRAIN_DIR+'/real')    
y_train_real = [1] * len(X_train_real)

X_train_edited = processImages(TRAIN_DIR+'/edited')    
y_train_edited = [0] * len(X_train_edited)

X_train = np.concatenate((X_train_real, X_train_edited), axis=0)
y_train = y_train_real + y_train_edited 

X_train = X_train.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 3)
print('X_train shape: {}'.format(X_train.shape))


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=SPLIT_RATE, random_state=1)


def create_classifier(dropout):
    classifier = Sequential() # Initialising the CNN    
    classifier.add(Conv2D(32, (3, 3), input_shape = (IMG_WIDTH, IMG_HEIGHT, 3), activation = 'relu')) 
    classifier.add(MaxPooling2D(pool_size = (2, 2))) 
    classifier.add(Dropout(dropout))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))  
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(dropout))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))  
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(dropout))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier


classifier = KerasClassifier(build_fn = create_classifier)

grid_parameters = {'epochs': N_EPOCHS,
                   'batch_size': BATCH_SIZES,
                   'dropout': DROPOUT_RATES                    
                  }


grid_search = GridSearchCV(estimator = classifier,
                           param_grid = grid_parameters,
                           scoring = 'accuracy',
                           cv = 4)


grid_search = grid_search.fit(X_train, y_train, verbose=2, validation_data = (X_val, y_val))

# %% [code]
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print('Best parameters: {}'.format(best_parameters))
print('Best accuracy: {:.4f}'.format(best_accuracy))


print('Confusion matrix for training set:')
train_predictions = grid_search.predict(X_train)
print(confusion_matrix(y_train, train_predictions))
print('\n')
print(classification_report(y_train, train_predictions))

print('Confusion matrix for validation set:')
val_predictions = grid_search.predict(X_val) 
print(confusion_matrix(y_val, val_predictions))
print('\n')
print(classification_report(y_val, val_predictions))


# Evaluating the model on unseen test images

X_test_real = processImages(TEST_DIR+'/real')    
y_test_real = [1] * len(X_test_real)

X_test_edited = processImages(TEST_DIR+'/edited')    
y_test_edited = [0] * len(X_test_edited)

X_test = np.concatenate((X_test_real, X_test_edited), axis=0)
y_test = y_test_real + y_test_edited 

X_test = X_test.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 3)
print('The dimensionality of the test dataset: {}'.format(X_test.shape))

print('Confusion matrix for test dataset:')
test_predictions = grid_search.predict(X_test)
print(confusion_matrix(y_test, test_predictions))
print('\n')
print(classification_report(y_test, test_predictions))
