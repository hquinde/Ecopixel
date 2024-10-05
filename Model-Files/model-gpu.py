import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop

imgWidth, imgHeight = 224, 224
trainDataDir = './full-dataset/TRAIN/'
validationDataDir = './full-dataset/TEST/'

## This part was aided by copiolet and stack overflow ##
# Model configuration
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(imgWidth, imgHeight, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=1e-5),
    metrics=['accuracy']
)

# Data preparation
trainDatagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
testDatagen = ImageDataGenerator(rescale=1. / 255)

trainGenerator = trainDatagen.flow_from_directory(trainDataDir, target_size=(imgWidth, imgHeight), batch_size=16, class_mode='categorical')
validationGenerator = testDatagen.flow_from_directory(validationDataDir, target_size=(imgWidth, imgHeight), batch_size=16, class_mode='categorical')
##################################

# Training the model
model.fit(trainGenerator, epochs=10, validation_data=validationGenerator)
model.save('model-wfd.h5')