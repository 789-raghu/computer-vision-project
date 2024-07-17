from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import base64
from pyngrok import ngrok

app = Flask(__name__)
model = load_model('/workspaces/computer-vision-project/model.h5')  # Adjust path to your model

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(225, 225)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def classify_image(img_path, model):
    input_image = preprocess_image(img_path)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions, axis=1)
    class_names = ['Healthy', 'Powdery', 'Rust']
    predicted_class_label = class_names[predicted_class_index[0]]
    return predicted_class_label, predictions[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global model

    if 'file' not in request.files:
        return jsonify({'result': "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'result': "No selected file"})

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join('/tmp', filename)
            file.save(file_path)

            predicted_class_label, probabilities = classify_image(file_path, model)

            original_img = cv2.imread(file_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            os.remove(file_path)

            img_str = cv2.imencode('.jpg', original_img)[1].tobytes()
            img_base64 = "data:image/jpeg;base64," + base64.b64encode(img_str).decode('utf-8')

            result_text = f"Healthy: {probabilities[0]:.2f}, Powdery: {probabilities[1]:.2f}, Rust: {probabilities[2]:.2f}"

            return render_template('result.html', prediction=predicted_class_label, probabilities=result_text, image=img_base64)

        except Exception as e:
            return jsonify({'result': f"Error processing file: {str(e)}"})

    else:
        return jsonify({'result': "File type not allowed"})

if __name__ == '__main__':
    ngrok.set_auth_token("2jMqFfEKUARor5aiD9gnjgUTJCK_6sXjLtNBZvvpecTbYiQSw")  # Replace with your ngrok auth token
    public_url = ngrok.connect(5000)
    print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:5000/\"".format(public_url))
    app.run()
