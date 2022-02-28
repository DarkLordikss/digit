from keras.models import load_model
import numpy as np
from PIL import Image

img = Image.open('test.png')
img.load()
model = load_model('mnist.h5')


def predict_digit(img):
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img/255.0
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

print(predict_digit(img))
