import numpy as np
import os
from flask import Flask, request, jsonify, render_template, url_for
import cv2
import numpy as np
import base64
import io
from PIL import Image

from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'C:/Users/Nageshreddy/Desktop/imagescount/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index2')
def home2():
    return render_template('index2.html')

@app.route('/predict1',methods=['POST'])
def predict1():

    print(request)
    f = request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
    image = cv2.imread(UPLOAD_FOLDER+f.filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(blur, 30, 100, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=0)

    (cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    im = Image.open(UPLOAD_FOLDER+f.filename)
    print("items in image ", len(cnt))
    data1 = io.BytesIO()
    im.save(data1, 'JPEG')
    prediction = len(cnt)
    encoded_img_data = base64.b64encode(data1.getvalue())

    return render_template('index.html', prediction_text1='Items in image {}'.format(prediction),
                           image_name1= f.filename, uploaded_image1=encoded_img_data.decode('utf-8'))


@app.route('/predict2',methods=['POST'])
def predict2():

    print(request)
    f = request.files['file']

    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
    img = cv2.imread(UPLOAD_FOLDER+f.filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY_INV)
    kernal = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(thresh, kernal, iterations=2)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = len(contours)
    im = Image.open(UPLOAD_FOLDER+f.filename)
    print("items in image ", objects)
    data1 = io.BytesIO()
    im.save(data1, 'JPEG')
    
    encoded_img_data = base64.b64encode(data1.getvalue())

    return render_template('index2.html', prediction_text2='Items in image {}'.format(objects),
                           image_name2= f.filename, uploaded_image2=encoded_img_data.decode('utf-8'))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        return render_template("success.html", name = f.filename)

if __name__ == "__main__":
    app.run(debug=True)
