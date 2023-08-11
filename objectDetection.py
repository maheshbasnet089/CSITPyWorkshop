import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load the pre-trained ResNet50 model (excluding top layers)
model = ResNet50(weights='imagenet', include_top=True)

# Function to preprocess image and predict objects
def predict_objects(image_path):
    image = cv2.imread(image_path)
    input_data = cv2.resize(image, (224, 224))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = preprocess_input(input_data)
    
    # Use the model to predict class probabilities
    predicted_probabilities = model.predict(input_data)
    decoded_predictions = decode_predictions(predicted_probabilities, top=3)[0]
    
    predicted_objects = [(class_name, score) for (_, class_name, score) in decoded_predictions]
    return predicted_objects

# Path to the image you want to predict
image_to_predict = 'fan2.jpg'

# Predict the objects in the image
predicted_objects = predict_objects(image_to_predict)
print("Predicted Objects:", predicted_objects)
