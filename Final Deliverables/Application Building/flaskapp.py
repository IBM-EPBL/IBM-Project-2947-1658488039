import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , render_template , request

app = Flask(__name__)
model = load_model('SignLanguageClasifier.h5')

@app.route('/')
def index:
    return render_template('index.html')
@app.route('/predict' , method = ['GET' , 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath , 'uploads' , f.filename)
        f.save(filepath)
        img = image.load.img(filepath , target_size = (64 , 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x , axis = 0)
        pred = mp.argmax(model.predict(x) , axis = 1)
        index = ['0' ,'1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',
                   'S','T','U','V','W','X','Y','Z']
        text = 'The Sign Langauge shown in the photo is : ' + str(index[pred[0]])
        return text

    if __name__ = '__main__':
        app.run()



