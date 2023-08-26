from keras.applications.resnet50 import (decode_predictions, preprocess_input)
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
keras = tf.keras


def load_model():
    '''load model'''
    loaded = tf.saved_model.load('pys/resnet50')
    infer = loaded.signatures['serving_default']
    return infer


def load_data(path: str):
    '''load data'''
    img = image.load_img(path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def format_result(result: list):
    '''format result'''
    # item ('n04310018', 'steam_locomotive', 0.6186888)
    return '\n'.join([f'{item[1]}: {round(item[2] * 100, 2)}%' for item in result])


def save_request(data: np.ndarray):
    '''save request'''
    with open('pys/request', 'wb') as f:
        f.write(data.tobytes())


if __name__ == '__main__':
    infer = load_model()
    img = load_data('pys/image/test.jpg')
    save_request(img)
    preds = infer(tf.constant(img))['predictions']
    top_3 = decode_predictions(preds.numpy(), top=3)[0]
    print(format_result(top_3))
