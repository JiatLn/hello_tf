from keras.applications.resnet50 import ResNet50
import tensorflow as tf
keras = tf.keras


# generate model
def gen_model():
    '''generate model'''
    model = ResNet50(weights='imagenet')
    model.save('pys/resnet50')


if __name__ == '__main__':
    gen_model()
