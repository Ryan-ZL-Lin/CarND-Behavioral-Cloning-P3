import csv
import os
from scipy import ndimage
import numpy as np
import math
import sklearn
from random import shuffle

lines = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

with open('recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
with open('left_curve/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
with open('left_curve2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
with open('right_curve/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
with open('bridge/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#the correction of left and right image on steering angle
correction = 0.5

### Use Generator to process data in batches
def generator(samples, batch_size=32):
    num_samples = len(samples)
    number_of_batches = num_samples/batch_size
    #print ('number_of_batches', number_of_batches)
    counter=0
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        if counter <= number_of_batches:
            #Produce batch data for training
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                measurements = []    
                for batch_sample in batch_samples:
                    steering_center = float(line[3])
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction
                    filename = ""
                    current_path = ""
                    
                    #center image and steering
                    source_path = line[0].strip()
                    if source_path.startswith('IMG'):
                        filename = source_path.split('/')[-1]
                        current_path = '/opt/carnd_p3/data/IMG/' + filename   
                    else:
                        current_path = source_path
                        
                    center_image = ndimage.imread(current_path)
                    images.append(center_image)
                    measurements.append(steering_center)

                    #left image and steering
                    source_path = line[1].strip()
                    if source_path.startswith('IMG'):
                        filename = source_path.split('/')[-1]
                        current_path = '/opt/carnd_p3/data/IMG/' + filename
                    else:
                        current_path = source_path
                        
                    left_image = ndimage.imread(current_path)
                    images.append(left_image)
                    measurements.append(steering_left)

                    #Data augmentation
                    image_flipped = np.fliplr(left_image)
                    measurement_flipped = -steering_left
                    images.append(image_flipped)
                    measurements.append(measurement_flipped)

                    #right image and steering
                    source_path = line[2].strip()
                    if source_path.startswith('IMG'):
                        filename = source_path.split('/')[-1]
                        current_path = '/opt/carnd_p3/data/IMG/' + filename
                    else:
                        current_path = source_path
                        
                    right_image = ndimage.imread(current_path)
                    images.append(right_image)
                    measurements.append(steering_right)

                    #Data augmentation
                    image_flipped = np.fliplr(right_image)
                    measurement_flipped = -steering_right
                    images.append(image_flipped)
                    measurements.append(measurement_flipped)

                x_train = np.array(images)
                y_train = np.array(measurements)
                
                yield sklearn.utils.shuffle(x_train, y_train)
                counter += 1
        
        if counter >= number_of_batches:
            counter = 0
                   
batch_size=32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

### Model Structure ###
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Lambda
import tensorflow as tf

model = Sequential()
# data normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# cropping data
model.add(Cropping2D(cropping=((70,25), (0,0))))

# Convolution Layers
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))


model.add(Flatten())
model.add(Dropout(0.3))
model.add(Activation('relu'))
          
# Dense Layers
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Activation('relu'))

model.add(Dense(1))

### model Structure End ###

print ('Training starts...')
from keras.models import Model
import matplotlib.pyplot as plt

model.compile(loss='mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch = math.ceil(len(train_samples)/batch_size),
                                     validation_data = validation_generator,
                                     validation_steps = math.ceil(len(validation_samples)/batch_size), 
                                     epochs=3, 
                                     verbose=1)

model.save("model.h5")
print ('Training ends...')


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig('loss_visualization.png')



exit()    

    
    

