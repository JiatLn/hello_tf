import io
from concurrent import futures

import grpc
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import decode_predictions, preprocess_input
from keras.preprocessing import image
from PIL import Image

import process_pb2_grpc
import process_pb2

keras = tf.keras


class Processer(process_pb2_grpc.ProcssServicer):
    def PreProcess(self, request, context):
        img = Image.open(io.BytesIO(request.image))
        img = img.convert('RGB')
        img = img.resize((224, 224), Image.NEAREST)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x: np.ndarray = preprocess_input(x)
        return process_pb2.PreProcessResponse(shape=x.shape, data=x.reshape(-1))

    def PostProcess(self, request, context):
        preds = np.array(request.data).reshape(
            request.shape).astype(np.float32)
        preds = decode_predictions(preds, top=3)[0]
        preds = [process_pb2.Preb(name=name, prob=prob)
                 for _, name, prob in preds]
        return process_pb2.PostProcessResponse(preds=preds)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
    process_pb2_grpc.add_ProcssServicer_to_server(Processer(), server)
    addr = '0.0.0.0:5002'
    server.add_insecure_port(addr)
    server.start()
    print('Starting server. Listening on ', addr)
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
