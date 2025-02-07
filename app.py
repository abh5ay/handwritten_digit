from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

app = Flask(__name__)
CORS(app)  


model = tf.keras.models.load_model("my_model0911.keras")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    
    try:
        img = Image.open(file).convert("L")  
        
        
        img_array = np.array(img)
        if np.mean(img_array) > 127:  
            img = ImageOps.invert(img)  
        
       
        img = img.resize((28, 28))
        
        
        img_array = np.array(img) / 255.0
        
       
        img_array = img_array.reshape(1, 28, 28, 1)

       
        prediction = model.predict(img_array).argmax()

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
