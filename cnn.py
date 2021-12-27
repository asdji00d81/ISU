from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
number_classes=2
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#classifier.add(layers.MaxPooling2D(pool_size = (3, 3)))
#classifier.add(layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#classifier.add(layers.MaxPooling2D(pool_size = (3, 3)))
classifier.add(layers.Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(layers.Conv2D(128, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(layers.Conv2D(128, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
#classifier.add(layers.MaxPooling2D(pool_size = (3, 3)))
#classifier.add(layers.Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
#classifier.add(layers.MaxPooling2D(pool_size = (3, 3)))

classifier.add(layers.Conv2D(128, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(layers.MaxPooling2D(pool_size = (3, 3)))

# Adding a second convolutional layer
# classifier.add(layers.Conv2D(128, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# classifier.add(layers.MaxPooling2D(pool_size = (3, 3)))

# classifier.add(layers.Conv2D(128, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# classifier.add(layers.MaxPooling2D(pool_size = (3, 3)))


# Step 3 - Flattening
classifier.add(layers.Flatten())
#classifier.add(layers.Dropout(0.2))

# Step 4 - Full connection
classifier.add(layers.Dense(units = 128, activation = 'relu'))
#classifier.add(Dense(units = 1, activation = 'relu'))
classifier.add(layers.Dense(number_classes, activation='sigmoid'))
#classifier.add(layers.Dense(units = 1, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/User/Desktop/2.dogcat_new - Copy/audiBMWdataset',
                                                 target_size = (64, 64),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('C:/Users/User/Desktop/2.dogcat_new - Copy/audiBMWdataset',
                                            target_size = (64, 64),
                                            batch_size = 16,
                                            class_mode = 'categorical')

model = classifier.fit(training_set,
                         steps_per_epoch = 16,
                         epochs = 20,
                         validation_data = test_set,    
                         validation_steps = 50)

classifier.save("model3.h5")
print("Saved model to disk")

classifier.summary()

# Part 3 - Making new predictions




