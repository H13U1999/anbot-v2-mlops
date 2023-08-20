import uuid
import requests
import PIL.Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


class DiscordBotNeuralStyleTransfer:
    def __init__(self):

        self.model = self.load_model()

    def load_model(self):
        hub_model = tf.keras.models.load_model('NTS-Model')

        return hub_model

    def combine_style(self, image_real, image_style):
        image_real = self.load_img(
            requests.get(image_real, stream=True).content)
        image_style = self.load_img(
            requests.get(image_style, stream=True).content)

        stylized_image = self.model(tf.constant(
            image_real), tf.constant(image_style))[0]
        rt_image = self.tensor_to_image(stylized_image)
        name = str(uuid.uuid1())+".jpg"
        rt_image.save(name)
        return name

    def tensor_to_image(self, tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def load_img(self, img):
        max_dim = 512
        # img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
