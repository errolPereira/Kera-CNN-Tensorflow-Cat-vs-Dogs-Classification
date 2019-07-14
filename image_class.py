# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:33:32 2019

@author: errol
"""
#importing packages
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#configuring the specification of model
from tensorflow.keras.optimizers import RMSprop

#Building a small convolution neural network
from tensorflow.keras import layers
from tensorflow.keras import Model

#Data Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras.preprocessing.image import array_to_img

#Extracting the zip files
#path
local_zip = 'E:/Data Science/Python/Imarticus LEarning/Tensorflow/scripts/CatVsDog/tmp/cats_and_dogs_filtered.zip'
#giving the file path to be extracted
zip_ref = zipfile.ZipFile(local_zip, 'r')
#extracting to the same folder
zip_ref.extractall('E:/Data Science/Python/Imarticus LEarning/Tensorflow/scripts/CatVsDog/tmp')
#closing the zip refrence
zip_ref.close()

#defining the training, testing and validation directory
#base directory
base_dir = 'E:/Data Science/Python/Imarticus LEarning/Tensorflow/scripts/CatVsDog/tmp/cats_and_dogs_filtered'
#train dir
train_dir = 'E:/Data Science/Python/Imarticus LEarning/Tensorflow/scripts/CatVsDog/tmp/cats_and_dogs_filtered/train'
#validation dir
validation_dir = 'E:/Data Science/Python/Imarticus LEarning/Tensorflow/scripts/CatVsDog/tmp/cats_and_dogs_filtered/validation'

#Directory with training cat pictures
train_cat_dir = os.path.join(train_dir, 'cats')
#Directory with training dogs pics
train_dog_dir = os.path.join(train_dir, 'dogs')

#Directory with validation cat pictures
validation_cat_dir = os.path.join(validation_dir, 'cats')
#Directory with validation dogs pics
validation_dog_dir = os.path.join(validation_dir, 'dogs')

#filenames in cats and dogs train directory
train_cat_frames = os.listdir(train_cat_dir)
print(train_cat_frames[:10])
train_dog_frames = os.listdir(train_dog_dir)
print(train_dog_frames[:10])


#total number of cats and dogs images in training and validation directory
print('Total train dog imgaes: {}'.format(len(os.listdir(train_dog_dir))))
print('Total validation dog imgaes: {}'.format(len(os.listdir(validation_dog_dir))))
print('Total train cat imgaes: {}'.format(len(os.listdir(train_cat_dir))))
print('Total validation cat imgaes: {}'.format(len(os.listdir(validation_cat_dir))))


#displaying a batch of 8 cats and 8 dogs
#parameters for the graph. Displaying the images as 4x4 
nrows = 4
ncols = 4
#index for iterating over the images
pic_index = 0

#set up matplotlib fig and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
pic_index += 8
next_cat_pic = [os.path.join(train_cat_dir, fname)
                for fname in train_cat_frames[pic_index-8:pic_index]]

next_dog_pic = [os.path.join(train_dog_dir, fname)
                for fname in train_dog_frames[pic_index-8:pic_index]]

for i, img_path in enumerate(next_cat_pic+next_dog_pic):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

####Convolutional Neural Net

#our input feature is of size 150x150x3, 150x150 is for the widht and height
# and 3 is for the color channels, R, G and B
img_input = layers.Input(shape=(150, 150, 3))

#First Convolution filter is of size 3x3 and extracts 16 filters
# followed by MaxPooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

#Second Convolution filter is of size 3x3 and extracts 32 filters
# followed by MaxPooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

#Third Convolution filter is of size 3x3 and extracts 64 filters
# followed by MaxPooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

#Flatten feature map to 1D vector so we can connect it to fully connected layers
x = layers.Flatten()(x)

#Fully connected layer with actiavtion function as relu and 512 hidden neurons
x = layers.Dense(512, activation='relu')(x)

#output layer with one node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

#Creating a model
model = Model(img_input, output)

#model summary
model.summary()



model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])


#Data PReprocessing
#Rescaling the images to 1/255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#flow training images in the batches of 20 using train_datagen
train_generator = train_datagen.flow_from_directory(
                    train_dir, #this is the source directory for the training images
                    target_size = (150, 150), #All images will be reshape to 150x150
                    batch_size=20,
                    class_mode='binary') #since we are using binary_crossentropy_loss


#flow training images in the batches of 20 using test_datagen
validation_generator = test_datagen.flow_from_directory(
                    validation_dir, #this is the source directory for the training images
                    target_size = (150, 150), #All images will be reshape to 150x150
                    batch_size=20,
                    class_mode='binary') #since we are using binary_crossentropy_loss


#training the model
history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 15,
        validation_data = validation_generator,
        validation_steps = 50,
        verbose = 2)




# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(img_input, successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
cat_img_files = [os.path.join(train_cat_dir, f) for f in train_cat_frames]
dog_img_files = [os.path.join(train_dog_dir, f) for f in train_dog_frames]
img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='BuGn_r')

#Evaluating accuracy of the model
#Retrieve list of accuracy results on training and test set for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']


#retrieving the loss for each training rpoch
loss = history.history['loss']
val_loss = history.history['val_loss']

#get the number of epochs
epochs = range(len(acc))

#plotting training and testing accuracy for each epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.legend(['Trainin accuracy', 'Testing Accuracy'])

plt.figure()

#plotting training and testing accuracy for each epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation accuracy')
plt.legend(['Trainin loss', 'Testing loss'])



##############################################################################################
############################     Dealing with overfitting      ###############################
##############################################################################################

#Creating a ImageDataGenerator and transforming the images
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#applying the datagen to a cat image to produce 5 variants of the image
img_path = os.path.join(train_cat_dir, train_cat_frames[500])
img = load_img(img_path, target_size=(150, 150))
x = img_to_array(img)
x = x.reshape((1,) + x.shape) #shape [1, 150, 150, 3]

# The .flow() command below generates batches of randomly transformed images
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    i += 1
    if i%5 == 0:
        break

#Adding data augmentation to preprocessing step
train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2)

#Note that the validation generator should not be augmented
test_datagen = ImageDataGenerator(rescale=1./255)

#flow training images in the batches of 20 using train_datagen
train_generator = train_datagen.flow_from_directory(
                    train_dir, #this is the source directory for the training images
                    target_size = (150, 150), #All images will be reshape to 150x150
                    batch_size=20,
                    class_mode='binary') #since we are using binary_crossentropy_loss


#flow training images in the batches of 20 using test_datagen
validation_generator = test_datagen.flow_from_directory(
                    validation_dir, #this is the source directory for the training images
                    target_size = (150, 150), #All images will be reshape to 150x150
                    batch_size=20,
                    class_mode='binary') #since we are using binary_crossentropy_loss

#Recreating the same model and adding dropout to overcome overfitting.
#input layer of shape 150x150x3
input_layer = layers.Input(shape=(150, 150, 3))

#First convolution layer with filter shape 3x3 and maxpooling shape 2x2 extracting 16 features
x = layers.Conv2D(16, 3, activation='relu')(input_layer)
x = layers.MaxPooling2D(2)(x)

#Second convolution layer with filter shape 3x3 and maxpooling shape 2x2 extracting 32 features
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

#Third convolution layer with filter shape 3x3 and maxpooling shape 2x2 extracting 64 features
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

#Flatten image to 1D tensor
x = layers.Flatten()(x)

#Fully connected dense layer 512 nodes
x = layers.Dense(512, activation='relu')(x)

#Adding dropout rate of 0.5
x = layers.Dropout(0.5)(x)

#output layer with sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

#configuring the model
model = Model(input_layer, output)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

model.summary()

#Retraining the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)


#Evaluating the results

# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')