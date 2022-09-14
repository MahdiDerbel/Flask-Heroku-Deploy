from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
import librosa
from flask import Flask
from flask_cors import CORS
from flask import jsonify
# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
#from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
CORS(app)
# Model saved with Keras model.save()
MODEL_PATH = "bestmodel2.hdf5"

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()   
model.summary()       # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
#Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('models/model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


#scaler.fit_transform(X_train)

def Predict_audio(audio_path,model):
        labels={'crépitant':0, 'NORMAL':1, 'ronflant':2, 'sibilant':3}
        audio, sample_rate = librosa.load(audio_path, res_type="kaiser_fast")

        feature = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)

        scaled_feature = np.mean(feature.T, axis=0)
        h=scaled_feature.reshape(1,16, 8, 1)
        labels = dict((v,k) for k,v in labels.items())
        predicted_conditions=model.predict(h)
        predicted_class_indices=np.argmax(predicted_conditions,axis=1)
        predictions = [labels[predicted_class_indices[0]]]
        return(predictions)



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request

        f = request.files["file"]
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prediction = Predict_audio(file_path, model)              # Convert to string
      
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
