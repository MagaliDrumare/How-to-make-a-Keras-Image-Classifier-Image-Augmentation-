# A voir et à savoir: 
* Augmenter les données permet d'éviter l'overfitting : https://fr.coursera.org/learn/deep-neural-network/lecture/Pa53F/other-regularization-methods
* https://keras.io/preprocessing/image/

```python
# Example of using .flow(x, y):

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
 ```
