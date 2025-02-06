from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model("my_model0911.keras")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    
    try:
        img = Image.open(file).convert("L")  # Convert image to grayscale
        
        # If the image has a white background, we might need to invert it
        img_array = np.array(img)
        if np.mean(img_array) > 127:  # Check if the image is light (white background)
            img = ImageOps.invert(img)  # Invert the image colors to make digits black on white
        
        # Resize to 28x28 (matching MNIST dimensions)
        img = img.resize((28, 28))
        
        # Normalize the image (values between 0 and 1)
        img_array = np.array(img) / 255.0
        
        # Reshape image to fit the model input (batch size, height, width, channels)
        img_array = img_array.reshape(1, 28, 28, 1)

        # Make the prediction
        prediction = model.predict(img_array).argmax()

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
