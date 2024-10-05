from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Path to image
imageLocation = "/Users/huntermimaroglu/Documents/Syracuse University/Clubs/CuseHacks/CuseHacks-2024/Test-Code/test/O_144.jpg"

# Load the model
model = load_model('/Users/huntermimaroglu/Documents/Syracuse University/Clubs/CuseHacks/CuseHacks-2024/Test-Code/model-v4.h5')

# Function that accepts an image and prepares it for the model's input
def preprocessImage(imagePath, targetSize=(224, 224)):
    img = load_img(imagePath, target_size=targetSize)
    imgArray = img_to_array(img)
    imgArray = imgArray / 255.0
    imgArray = np.expand_dims(imgArray, axis=0)
    return imgArray

# Function to process an image and predict its class
def processImage(imagePath):
    processedImage = preprocessImage(imagePath)
    output = model.predict(processedImage)
    classLabels = ['NON-RECYCLABLE', 'ORGANIC', 'RECYCLABLE']  
    imageType = classLabels[np.argmax(output)]    
    return imageType

# Outputting the type to a .txt file
file = open("trashType.txt", "w")  
file.write(processImage(imageLocation))
file.close()