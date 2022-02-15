import os
import numpy as np

from keras.models import  load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2

UPLOAD_FOLDER = 'Uploads'

# Define a flask app
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def model_predict(img_path):
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(224,224))
    img=np.array(img)
    img=img.reshape((-1,img.shape[0],img.shape[1],img.shape[2]))
    basepath = os.path.dirname(__file__)
    model_path = os.path.join(
        basepath, 'model.h5')

    mymodel = load_model(model_path)
    y = mymodel.predict(img)
    if (int(y))==0:
        return ("Rugby")
    else:
        return ("Soccer")

    return pred


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'Uploads', secure_filename(f.filename))
        f.save(file_path)

        result = model_predict(file_path)
        return (result)
    return None


if __name__ == '__main__':
    app.run(debug=True)

