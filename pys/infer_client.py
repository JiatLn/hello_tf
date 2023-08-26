import grpc
import infer_pb2_grpc
import infer_pb2
from keras.applications.resnet50 import (decode_predictions, preprocess_input)
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
keras = tf.keras


def load_data(path: str) -> np.ndarray:
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


def run_infer(img: np.ndarray, addr='localhost:5000') -> list:
    with grpc.insecure_channel(addr) as chan:
        stub = infer_pb2_grpc.InferStub(chan)
        shape, data = img.shape, img.reshape(-1)
        request = infer_pb2.InferRequest(shape=shape, data=data)
        res = stub.Infer(request)
        preds = np.array(res.data).reshape(res.shape)
        top_3 = decode_predictions(preds, top=3)[0]
        return top_3


if __name__ == '__main__':
    img = load_data('pys/image/test.jpg')
    res = run_infer(img)
    print(format_result(res))
