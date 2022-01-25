from cgitb import handler
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from logging import FileHandler, WARNING
import numpy as np
import os
import cv2

from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model_corn = load_model('corn_model.h5')

# class_dict = {'1ha,0,0,0': 'Blight', '0,0,0,1': 'Healthy', '0,0,1,0':'Gray_Leaf_Spot', '0,1,0,0':'Common_Rust'}

def predict_label(img_path, method):
    # query = cv2.imread(img_path)
    # output = query.copy()
    # query = cv2.resize(query, (32, 32))
    # q = []
    # q.append(query)
    # q = np.array(q, dtype='float') / 255.0
    # q_pred = model_corn.predict(q)
    # return q_pred

    img = image.load_img(img_path, target_size=(128,128))
    # imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    model = load_model('corn_model.h5')
    classes = model.predict(images, batch_size=10)
    
    #   print(fn)
    # print('classes', classes)
    if classes[0, 0]:
        return 'Blight'
    elif classes[0, 1]:
        return 'Common Rust'
    elif classes[0, 2]:
        return 'Gray Leaf Spot'
    elif Classes[0, 3]:
        return 'Healthy'

    # if (method == "corn_disease"):
    #     q_pred = model_corn.predict(q)
    # else:
    #     return 'error: invalid method name'
    # if q_pred<=0.5 :
    #     predicted_bit = 0
    # else :
    #     predicted_bit = 1
    # return class_dict[predicted_bit]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/corn_disease', methods=['GET', 'POST'])
def corn_disease():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path, 'corn_disease')
            return render_template('corn_disease.html', uploaded_image=image.filename, prediction=prediction)
    return render_template('corn_disease.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)