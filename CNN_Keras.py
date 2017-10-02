#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@Credit : deep Learning A-Z Hands-On Artificial Neural Networks 
Kirill Eremenko 
Hedelin de Ponteves 
Created on Mon Oct  2 16:03:51 2017
@author: magalidrumare
"""

# Upload the Data : http://www.superdatascience.com/wp-content/uploads/2017/04/Convolutional_Neural_Networks.zip 

#Part 1 _Image pre-processing 
# 10000 images 
# 8000 images dans training set dogs 
# 4000 each for cats and dogs 
# 2000  images dans le test set 
# 1000 each for cats and dogs 

#Importer les Keras packages to make the CNN 
from keras.models import Sequential # initialiser le neural network 
from keras.layers import Convolution2D # convolutional step 
from keras.layers import MaxPooling2D # max pooling step 
from keras.layers import Flatten # convert all the features maps into an large feature vector = input du Full Connected layer
from keras.layers import Dense # add the fully connected layer 
#using the TensorFlow Backend 


# Initialize the CNN in keras 
classifier= Sequential() 

#CNN step by step  Convolution > Relu> Maxpooling>Flattening> Full Connection 
# Step 1 _ Convolution 
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))
## 32 features detector of 3x3 pixels 
## input_shape : 
## 3 = colour channel 
## format of the images is 64*64 px 
## activation function to obtain non linearity 

# Step 2_Maxpooling (reduce the size of the feature maps)
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Adding a second convolutional layer
classifier.add(Convolution2D(32,3,3,activation='relu')) # Keras récupère automatiquement de input_shape
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3_Flattening 
classifier.add(Flatten()) # all the feature maps are converted in one flatten vector

#Full Connexion with the hidden layer 
classifier.add(Dense(output_dim=128, activation='relu')) 
# output_dim = number of hidden layers 

# Full Connexion with the output layer 
classifier.add(Dense(output_dim=1, activation='sigmoid')) 
# sigmoid because the result will be cat or dog. 
#output_dim = a unique label of the image cat or dog.

#Compiling the CNN 
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# adam optimizer is one of the optimizer for the gradient descent 
# binary_cross entropy = result binaire cat or dogs 

# Image Augmentation thans Keras documentation 
# to avoid ovefitting. Good result on the training set and por result on the test set 
# https://keras.io/preprocessing/image/
# Example of using .flow_from_directory(directory):
    
from keras.preprocessing.image import ImageDataGenerator # importer ImageDataGenerator 

train_datagen = ImageDataGenerator( # augmentation of the images. 
        rescale=1./255, # normalization of the images 
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) # normalization of the images 

training_set = train_datagen.flow_from_directory(
        'dataset/training_set', # url of the trainig_set file
        target_size=(64,64), # size of the input images 
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set', # url of the test_set file
        target_size=(64,64), #size of the input images 
        batch_size=32,
        class_mode='binary')

# Train the Classifier 
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, # 8000 images in the training_set 
        epochs=20,
        validation_data=test_set,
        validation_steps=2000) # 2000 images in the test_set 


#Making new Predictions 
import numpy as np
from keras.preprocessing import image 
test_image=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
# new dimension thrird dimension to the image 
test_image=image.img_to_array(test_image)
# fourth dimension which will be the batch 
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result [0][0] == 1 :
    prediction ='dog' 
else : 
  prediction ='cat' 



