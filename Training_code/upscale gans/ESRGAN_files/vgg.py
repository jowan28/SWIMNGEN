# import the necessary packages
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model


class VGG:
    @staticmethod
    def build():
        # initialize the pre-trained VGG19 model
        vgg = VGG19(input_shape=(None, None, 3), weights="imagenet", include_top=False)
        # slicing the VGG19 model till layer #20
        model = Model(vgg.input, vgg.layers[20].output)
        # return the sliced VGG19 model
        return model
