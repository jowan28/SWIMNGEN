import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from PIL import Image
from tensorflow.keras import layers
import keras
import time
import tensorflow as tf
import h5py
import skimage
import sys
from IPython import display
import csv
from keras.utils.vis_utils import plot_model
from keras import backend
import absl.logging
from tensorflow.keras.layers import Layer, InputSpec

absl.logging.set_verbosity(absl.logging.ERROR)


# weighted sum output
class WeightedSum(keras.layers.Add):
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = keras.backend.variable(alpha, name="ws_alpha")

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert len(inputs) == 2
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


def one_hot_generator_input_layer(n_classes, latent_dim):
    latent_input = tf.keras.Input(shape=(latent_dim,))
    label_input = tf.keras.Input(shape=(n_classes,))

    concatenated = layers.Concatenate(axis=1)([latent_input, label_input])
    concatenated = layers.Dense(128 * 8 * 8)(concatenated)
    concatenated = layers.ReLU()(concatenated)
    concatenated = layers.Reshape((8, 8, 128))(concatenated)
    concatenated = layers.Conv2DTranspose(128, (3, 3), padding="same")(concatenated)
    embedded_model = tf.keras.Model([latent_input, label_input], concatenated)
    embedded_model._name = "embedding"
    return embedded_model


def generator_block(res):
    block = tf.keras.Sequential()
    block.add(layers.BatchNormalization(input_shape=(res, res, 128)))
    # input resolution: ir
    block.add(layers.LeakyReLU())
    block.add(layers.UpSampling2D())
    block.add(layers.Conv2DTranspose(128, (4, 4), padding="same"))
    block._name = "block_ir_" + str(res)
    return block


def generator_output_layer(res):
    output_layer = tf.keras.Sequential()
    output_layer.add(
        layers.Conv2D(4, (1, 1), input_shape=(res, res, 128), activation=tf.nn.tanh)
    )
    output_layer.add(layers.Lambda(lambda x: 0.5 * x + 0.5))
    output_layer._name = "output_layer"
    return output_layer


def conditional_discriminator_input_layer(res, n_classes):
    label_input = tf.keras.Input(shape=(n_classes,))
    label = layers.Dense(n_classes * res * res)(label_input)
    label = layers.Reshape((res, res, n_classes))(label)

    image_input = tf.keras.Input(shape=(res, res, 4))
    concatenated = layers.Concatenate()([image_input, label])
    concatenated = layers.Conv2D(64, (1, 1), padding="same")(concatenated)
    concatenated = layers.LeakyReLU()(concatenated)
    embedded_model = tf.keras.Model([image_input, label_input], concatenated)
    embedded_model._name = "embedding"
    return embedded_model


def discriminator_block(res):
    block = tf.keras.Sequential()
    block.add(layers.Conv2D(64, (3, 3), padding="same", input_shape=(res, res, 64)))
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())
    block.add(layers.Dropout(0.4))
    block.add(layers.AveragePooling2D())
    block._name = "block_ir_" + str(res)
    return block


def discriminator_output_layer():
    output_layer = tf.keras.Sequential()
    output_layer.add(layers.Conv2D(64, (4, 4), input_shape=(4, 4, 64), padding="same"))
    output_layer.add(layers.Dropout(0.6))
    output_layer.add(layers.Conv2D(32, (3, 3)))
    output_layer.add(layers.Flatten())
    # removed the sigmoid activation layer to allow logits for binary cross entropy
    output_layer.add(layers.Dense(1))
    output_layer._name = "output_layer"
    return output_layer


def final_generator_block(output_resolution):
    # input size
    input_res = int(2 ** np.floor(np.log2(output_resolution)))
    block = tf.keras.Sequential()
    block.add(layers.BatchNormalization(input_shape=(input_res, input_res, 128)))
    block.add(layers.LeakyReLU())
    block.add(
        layers.Lambda(
            lambda x: tf.image.resize(x, [output_resolution, output_resolution])
        )
    )
    block.add(layers.Conv2DTranspose(128, (4, 4), padding="same"))
    block._name = "block_ir_" + str(input_res)
    return block


def final_discriminator_block(input_resolution):
    output_resolution = int(2 ** np.floor(np.log2(input_resolution)))
    block = tf.keras.Sequential()
    block.add(
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            input_shape=(input_resolution, input_resolution, 64),
        )
    )
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())
    block.add(layers.Dropout(0.4))
    block.add(
        layers.Lambda(
            lambda x: tf.image.resize(x, [output_resolution, output_resolution])
        )
    )
    block._name = "block_ir_" + str(input_resolution)
    return block


class Attention(Layer):
    def __init__(self, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        print(kernel_shape_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(
            name="gamma", shape=[1], initializer="zeros", trainable=True
        )
        self.kernel_f = self.add_weight(
            shape=kernel_shape_f_g, initializer="glorot_uniform", name="kernel_f"
        )
        self.kernel_g = self.add_weight(
            shape=kernel_shape_f_g, initializer="glorot_uniform", name="kernel_g"
        )
        self.kernel_h = self.add_weight(
            shape=kernel_shape_h, initializer="glorot_uniform", name="kernel_h"
        )
        self.bias_f = self.add_weight(
            shape=(self.filters_f_g,), initializer="zeros", name="bias_F"
        )
        self.bias_g = self.add_weight(
            shape=(self.filters_f_g,), initializer="zeros", name="bias_g"
        )
        self.bias_h = self.add_weight(
            shape=(self.filters_h,), initializer="zeros", name="bias_h"
        )
        super(Attention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={3: input_shape[-1]})
        self.built = True

    def call(self, x):
        def hw_flatten(x):
            return backend.reshape(
                x,
                shape=[
                    backend.shape(x)[0],
                    backend.shape(x)[1] * backend.shape(x)[2],
                    backend.shape(x)[-1],
                ],
            )

        f = backend.conv2d(
            x, kernel=self.kernel_f, strides=(1, 1), padding="same"
        )  # [bs, h, w, c']
        f = backend.bias_add(f, self.bias_f)
        g = backend.conv2d(
            x, kernel=self.kernel_g, strides=(1, 1), padding="same"
        )  # [bs, h, w, c']
        g = backend.bias_add(g, self.bias_g)
        h = backend.conv2d(
            x, kernel=self.kernel_h, strides=(1, 1), padding="same"
        )  # [bs, h, w, c]
        h = backend.bias_add(h, self.bias_h)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = backend.softmax(s, axis=-1)  # attention map

        o = backend.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = backend.reshape(o, shape=backend.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class conditional_generator(tf.keras.Model):
    def __init__(
        self, resolution, model="not yet defined", classes=10, latent_shape=100
    ):
        super(conditional_generator, self).__init__()
        self.classes = classes
        self.resolution = resolution
        self.model = model
        self.latent_shape = latent_shape

    def summary(self):
        self.model.summary()

    def display(self):
        plot_model(self.model, show_shapes=True, show_layer_names=True)

    def input_layer(self):
        # create the input layer from latent space and the label
        input_layer = one_hot_generator_input_layer(self.classes, self.latent_shape)
        self.model = input_layer

    def block(self):
        # create block to add to the model
        # this block is not branched
        res = self.model.output_shape[1]
        block = generator_block(res)
        self.model.add(block)

    def output_layer(self):
        res = self.model.output_shape[1]
        output_layer = generator_output_layer(res)
        self.model.add(output_layer)

    def build_model(self):
        res = self.resolution
        if (res & (res - 1) == 0) and res != 0:
            self.input_layer()
            latent_input = tf.keras.Input(shape=(self.latent_shape,))
            label_input = tf.keras.Input(shape=(self.classes,))
            concatenated = self.model([latent_input, label_input])
            while concatenated.shape[1] != res:
                concatenated = generator_block(concatenated.shape[1])(concatenated)
            output = generator_output_layer(concatenated.shape[1])(concatenated)
            self.model = tf.keras.Model([latent_input, label_input], output)
        else:
            print("not a power of 2")

    def grow(self, trainable_var):
        # must ensure that we do not grow on leaves (RGB layer)
        label_input = tf.keras.Input(shape=(self.classes,))
        latent_input = tf.keras.Input(shape=(self.latent_shape,))
        # start with the embedder
        roots = self.model.layers[2]
        roots.trainable = trainable_var
        roots._name = "roots"
        climber = roots([latent_input, label_input])
        for layer in self.model.layers[3:-1]:
            layer.trainable = trainable_var
            climber = layer(climber)

        # trim the current output layer
        if isinstance(self.model.layers[-1].layers[-1], keras.Sequential):
            twigs = tf.keras.Sequential(self.model.layers[-1].layers[:-1])
            twigs._name = "block_ir_" + str(twigs.input_shape[1])
            twigs.trainable = trainable_var
            climber = twigs(climber)
        elif isinstance(self.model.layers[-1].layers[-1], layers.Lambda):
            pass
        else:
            twigs = tf.keras.Sequential(self.model.layers[-1].layers)
            twigs._name = "block_ir_" + str(twigs.input_shape[1])
            twigs.trainable = trainable_var
            climber = twigs(climber)

        model_output = climber
        # the branches that grow and die
        res = self.model.output_shape[1]
        growing_branch = generator_block(res)
        growing_branch.add(generator_output_layer(2 * res))
        growing_branch._name = "growing_branch"

        # converts to rgb before upscaling
        dying_branch = tf.keras.Sequential()
        convlayer = layers.Conv2D(
            4, (1, 1), input_shape=(res, res, 128), activation=tf.nn.tanh
        )
        convlayer.build(input_shape=(res, res, 128))
        # use old model output layer to preserve the images
        # to allow for new models, not just to continue training
        if isinstance(self.model.layers[-1].layers[1], layers.Lambda):
            rgb_weights = self.model.layers[-1].layers[0].weights
        else:
            rgb_weights = self.model.layers[-1].layers[-1].layers[0].weights
        convlayer.set_weights(rgb_weights)
        dying_branch.add(convlayer)
        dying_branch.add(layers.Lambda(lambda x: 0.5 * x + 0.5))
        dying_branch.add(layers.UpSampling2D(input_shape=(res, res, 128)))
        dying_branch._name = "dying_branch"

        dying_out = dying_branch(model_output)
        growing_out = growing_branch(model_output)

        canopy = WeightedSum(name="canopy")([dying_out, growing_out])

        self.model = keras.Model([latent_input, label_input], canopy)
        self.resolution = self.model.output_shape[1]

    def update_fadein(self, step, n_steps):
        # calculate current alpha (linear from 0 to 1)
        alpha = step / float(n_steps - 1)
        # update the alpha for each model
        for layer in self.model.layers:
            if isinstance(layer, WeightedSum):
                tf.keras.backend.set_value(layer.alpha, alpha)
        # update the alpha value on each instance of WeightedSum
        # the weighted sum is always the last layer

    def prune(self, trainable_var):
        # this removes the dying branch
        # get the penultimate element (healthy branch)
        healthy = self.model.layers[-2]
        healthy._name = "block_ir_" + str(healthy.input_shape[1])
        # get the trunk
        label_input = tf.keras.Input(shape=(self.classes,))
        latent_input = tf.keras.Input(shape=(self.latent_shape,))
        # start with the embedder
        roots = self.model.layers[2]
        roots.trainable = trainable_var
        roots._name = "roots"
        climber = roots([latent_input, label_input])
        for layer in self.model.layers[3:-3]:
            layer.trainable = trainable_var
            climber = layer(climber)
        grown = climber
        fully_grown = healthy(grown)
        self.model = tf.keras.Model(
            [latent_input, label_input], fully_grown, name="grown"
        )

    def final_growth(self, output_resolution, trainable_var):
        # must ensure that we do not grow on leaves (RGB layer)
        label_input = tf.keras.Input(shape=(self.classes,))
        latent_input = tf.keras.Input(shape=(self.latent_shape,))
        # start with the embedder
        roots = self.model.layers[2]
        roots.trainable = trainable_var
        roots._name = "roots"
        climber = roots([latent_input, label_input])
        for layer in self.model.layers[3:-1]:
            layer.trainable = trainable_var
            climber = layer(climber)

        # trim the current output layer
        if isinstance(self.model.layers[-1].layers[-1], keras.Sequential):
            twigs = tf.keras.Sequential(self.model.layers[-1].layers[:-1])
            twigs._name = "block_ir_" + str(twigs.input_shape[1])
            twigs.trainable = trainable_var
            climber = twigs(climber)
        elif isinstance(self.model.layers[-1].layers[-1], layers.Lambda):
            pass
        else:
            twigs = tf.keras.Sequential(self.model.layers[-1].layers)
            twigs._name = "block_ir_" + str(twigs.input_shape[1])
            twigs.trainable = trainable_var
            climber = twigs(climber)

        model_output = climber
        # the branches that grow and die
        res = self.model.output_shape[1]
        growing_branch = final_generator_block(output_resolution)
        growing_branch.add(generator_output_layer(output_resolution))
        growing_branch._name = "growing_branch"

        # converts to rgb before upscaling
        dying_branch = tf.keras.Sequential()
        convlayer = layers.Conv2D(
            4, (1, 1), input_shape=(res, res, 128), activation=tf.nn.tanh
        )
        convlayer.build(input_shape=(res, res, 128))
        # use old model output layer to preserve the images
        rgb_weights = self.model.layers[-1].layers[-1].layers[0].weights
        convlayer.set_weights(rgb_weights)
        dying_branch.add(convlayer)
        dying_branch.add(layers.Lambda(lambda x: 0.5 * x + 0.5))
        dying_branch.add(
            layers.Lambda(
                lambda x: tf.image.resize(x, [output_resolution, output_resolution])
            )
        )
        dying_branch._name = "dying_branch"

        dying_out = dying_branch(model_output)
        growing_out = growing_branch(model_output)

        canopy = WeightedSum(name="canopy")([dying_out, growing_out])

        self.model = keras.Model([latent_input, label_input], canopy)
        self.resolution = self.model.output_shape[1]


class conditional_discriminator(tf.keras.Model):
    def __init__(
        self, resolution, model="not defined yet", classes=10, embedding_dim=100
    ):
        super(conditional_discriminator, self).__init__()
        self.model = model
        self.classes = classes
        self.resolution = resolution
        self.embedding_dim = embedding_dim

    def summary(self):
        self.model.summary()

    def display(self):
        plot_model(self.model, show_shapes=True, show_layer_names=True)

    def input_layer(self):
        # create the input layer for the image
        res = self.resolution
        input_layer = conditional_discriminator_input_layer(res, self.classes)
        self.model = input_layer

    def block(self):
        # create block to add to the model
        # this block is not branched
        res = self.model.output_shape[1]
        block = discriminator_block(res)
        self.model.add(block)

    def output_layer(self):
        res = self.model.output_shape[1]
        output_layer = discriminator_output_layer()
        self.model.add(output_layer)

    def build_model(self):
        res = self.resolution
        if (res & (res - 1) == 0) and res != 0:
            self.input_layer()
            label_input = tf.keras.Input(shape=(self.classes,))
            image_input = tf.keras.Input(shape=(self.resolution, self.resolution, 4,))
            concatenated = self.model([image_input, label_input])
            # add a self attention layer after the first conv layer

            while concatenated.shape[1] != 4:
                concatenated = discriminator_block(concatenated.shape[1])(concatenated)

            attention_layer = Attention(concatenated.shape[-1])
            attention_layer.build(concatenated.shape)
            concatenated = attention_layer(concatenated)
            output = discriminator_output_layer()(concatenated)
            self.model = tf.keras.Model([image_input, label_input], output)
        else:
            print("not a power of 2")

    def grow(self, trainable_var):
        # must ensure that we do not grow on leaves (RGB layer)
        resolution = self.model.input_shape[0][1]
        # the input
        label_input = tf.keras.Input(shape=(self.classes,))
        full_image_input = tf.keras.Input(shape=(2 * resolution, 2 * resolution, 4,))
        full_encoder = conditional_discriminator_input_layer(
            2 * resolution, self.classes
        )
        full_encoder._name = "input_encoder"
        half_image_input = layers.AveragePooling2D()(full_image_input)
        half_encoder = self.model.layers[2]
        half_encoder._name = "dying_encoder"

        growing_branch = discriminator_block(2 * resolution)
        growing_branch._name = "block_ir_" + str(2 * resolution)

        growing_in = full_encoder([full_image_input, label_input])
        growing_in = growing_branch(growing_in)

        dying_in = half_encoder([half_image_input, label_input])

        fork = WeightedSum(name="fork")([dying_in, growing_in])
        descender = fork

        # skip the input rgb
        # it goes from 4 not 3 so that the top two layers are trained
        for index, layer in enumerate(self.model.layers[3:]):
            if index == 0:
                trainable_var_des = True
            else:
                trainable_var_des = trainable_var
            layer.trainable = trainable_var_des
            descender = layer(descender)

        model_output = descender
        self.model = keras.Model([full_image_input, label_input], model_output)
        self.resolution = self.model.input_shape[0][1]

    def final_growth(self, input_resolution, trainable_var):
        intermediate_resolution = int(2 ** np.floor(np.log2(input_resolution)))
        # must ensure that we do not grow on leaves (RGB layer)
        resolution = self.model.input_shape[0][1]
        # the input
        label_input = tf.keras.Input(shape=(self.classes,))
        full_image_input = tf.keras.Input(
            shape=(input_resolution, input_resolution, 4,)
        )
        full_encoder = conditional_discriminator_input_layer(
            input_resolution, self.classes
        )
        full_encoder._name = "input_encoder"
        half_image_input = layers.Lambda(
            lambda x: tf.image.resize(
                x, [intermediate_resolution, intermediate_resolution]
            )
        )(full_image_input)
        half_encoder = self.model.layers[2]
        half_encoder._name = "dying_encoder"

        growing_branch = final_discriminator_block(input_resolution)
        growing_branch._name = "block_ir_" + str(input_resolution)

        growing_in = full_encoder([full_image_input, label_input])
        growing_in = growing_branch(growing_in)

        dying_in = half_encoder([half_image_input, label_input])

        fork = WeightedSum(name="fork")([dying_in, growing_in])
        descender = fork

        # skip the input rgb
        for layer in self.model.layers[3:]:
            layer.trainable = trainable_var
            descender = layer(descender)

        model_output = descender
        self.model = keras.Model([full_image_input, label_input], model_output)
        self.resolution = self.model.input_shape[0][1]

    def update_fadein(self, step, n_steps):
        # calculate current alpha (linear from 0 to 1)
        alpha = step / float(n_steps - 1)
        # update the alpha for each model
        for layer in self.model.layers:
            if isinstance(layer, WeightedSum):
                tf.keras.backend.set_value(layer.alpha, alpha)

        # update the alpha value on each instance of WeightedSum
        # the weighted sum is always the last layer

    def prune(self, trainable_var):
        # this removes the dying branch
        # get the penultimate element (healthy branch)
        label_input = tf.keras.Input(shape=(self.classes,))
        image_input = tf.keras.Input(shape=(self.resolution, self.resolution, 4,))
        if isinstance(
            self.model.layers[2], keras.engine.functional.Functional
        ) and isinstance(self.model.layers[4], keras.engine.sequential.Sequential):
            healthy_encoder = self.model.layers[2]
            healthy_block = self.model.layers[4]
        else:
            healthy_encoder = self.model.layers[3]
            healthy_block = self.model.layers[5]
        descender = healthy_encoder([image_input, label_input])
        descender = healthy_block(descender)
        # get the trunk
        if isinstance(self.model.layers[7], layers.Add):
            for layer in self.model.layers[8:]:
                layer.trainable = trainable_var
                descender = layer(descender)
        else:
            for layer in self.model.layers[7:]:
                layer.trainable = trainable_var
                descender = layer(descender)
        fully_grown = descender
        self.model = tf.keras.Model(
            [image_input, label_input], fully_grown, name="grown"
        )


class dataset:
    def __init__(self, var="not defined"):
        self.var = var

    # this is focused on easily getting data and formatting it
    # datasets sized so can iterate through it in an epoch
    def get_data(self, resolution, BATCH_SIZE):
        # this creates a tf suitable 'matirx' of data which can be iterated through
        # start with just swimseg
        def LoadData(path1):
            """
            Looks for relevant filenames in the shared path
            Returns 2 lists for original and masked files respectively

            """
            # Read the images folder like a list
            image_dataset = os.listdir(path1)

            # Make a list for images and masks filenames
            orig_img = []
            for file in image_dataset:
                orig_img.append(file)

            # Sort the lists to get both of them in same order (the dataset has exactly the same name for images and corresponding masks)
            orig_img.sort()

            return orig_img

        def PreprocessData(img, target_shape_img, path1):
            """
            Processes the images and mask present in the shared list and path
            Returns a NumPy dataset with images as 3-D arrays of desired size
            Please note the masks in this dataset have only one channel
            """
            img_path = path1 + "images/"
            map_path = path1 + "GTmaps/"
            # Pull the relevant dimensions for image and mask
            m = len(img)  # number of images
            (
                i_h,
                i_w,
                i_c,
            ) = target_shape_img  # pull height, width, and channels of image

            # Define X and Y as number of images along with shape of one image
            Z = np.zeros((m, i_h, i_w, i_c + 1), dtype=np.float32)
            # Resize images and masks
            for file in img:
                # convert image into an array of desired shape (3 channels)
                index = img.index(file)
                i_path = os.path.join(img_path, file)
                m_path = os.path.join(map_path, file)
                single_img = Image.open(i_path).convert("RGB")
                single_img = single_img.resize((i_h, i_w))
                single_img = np.reshape(single_img, (i_h, i_w, i_c))
                single_img = single_img / 255.0
                single_map = Image.open(m_path)
                single_map = single_map.resize((i_h, i_w))
                single_map = np.reshape(single_map, (i_h, i_w, 1))
                Z[index][:, :, :i_c] = single_img
                Z[index][:, :, i_c:] = single_map
            return Z

        def scale_dataset(target_shape, path="/home2/vcqf59/Large_aug_swimseg/"):
            img = LoadData(path + "images/")
            # Process data using apt helper function
            Z = PreprocessData(img, target_shape, path)
            return Z

        save_file = (
            "/home2/vcqf59/Classified/masters-project/dataset pickle/half_swimnseg_"
            + str(resolution)
            + "_pickled_dataset"
        )
        # check to see if folder exists
        if os.path.isdir(save_file) == False:
            swimseg = scale_dataset((resolution, resolution, 3))
            swimseg_label = [1 for i in swimseg]
            swinseg = scale_dataset(
                (resolution, resolution, 3), path="/home2/vcqf59/Large_aug_swinseg/"
            )
            swinseg_label = [0 for i in swinseg]
            data = np.append(swimseg, swinseg, 0)
            labels = np.append(swimseg_label, swinseg_label, 0)
            real = [1 for i in data]

            # else:
            # self.batches =
            BUFFER_SIZE = int(np.max(np.shape(swimseg))) + int(
                np.max(np.shape(swinseg))
            )
            dataset = (
                tf.data.Dataset.from_tensor_slices((data, labels))
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
            )
            self.train_dataset = dataset
        else:
            self.train_dataset = tf.data.Dataset.load(save_file)


class ConditionalGAN(keras.Model):
    def __init__(
        self,
        resolution,
        latent_dim,
        classes=2,
        final_resolution="not defined",
        csv_log="not_defined",
        update_threshold=0,
        noise_factor=0,
        transfer=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.discriminator = conditional_discriminator(
            resolution, classes=classes, embedding_dim=latent_dim
        )
        self.generator = conditional_generator(
            resolution, classes=classes, latent_shape=latent_dim
        )
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.final_resolution = final_resolution
        self.csv_log = csv_log
        self.update_threshold = update_threshold
        self.noise_factor = noise_factor
        self.classes = int(classes)
        self.initial_resolution = resolution
        self.transfer = not (transfer)
        self.initial_alpha = 0

    def build(self):
        self.discriminator.build_model()
        self.generator.build_model()

    def grow(self):
        self.discriminator.grow(self.transfer)
        self.generator.grow(self.transfer)
        self.resolution = self.resolution * 2

    def final_growth(self):
        self.discriminator.final_growth(self.final_resolution, self.transfer)
        self.generator.final_growth(self.final_resolution, self.transfer)
        self.resolution = self.final_resolution

    def prune(self):
        self.discriminator.prune(self.transfer)
        self.generator.prune(self.transfer)

    def fadein(self):
        self.discriminator.update_fadein(self.step, self.fadein_steps)
        self.generator.update_fadein(self.step, self.fadein_steps)

    def batch_set(self):
        if self.resolution > 100:
            self.batch_size = 32
        elif self.resolution > 130:
            self.batch_size = 16
        else:
            self.batch_size = 32

    def epoch_set(self):
        if self.resolution == 8:
            self.train_steps = 8000
            self.fadein_steps = 2000
        elif self.resolution == 16:
            self.train_steps = 8000
            self.fadein_steps = 2000
        elif self.resolution == 32:
            self.train_steps = 8000
            self.fadein_steps = 3000
        elif self.resolution == 64:
            self.train_steps = 10000
            self.fadein_steps = 3000
        else:
            self.train_steps = 12000
            self.fadein_steps = 4000

        self.generator.n_steps = self.fadein_steps
        self.discriminator.n_steps = self.fadein_steps

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile(run_eagerly=True)
        # this is run eagerly so that the tensor iterators can be split in the train_step function
        self.discriminator_optimizer = d_optimizer
        self.generator_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def add_gaussian_noise(self, image):
        # image must be scaled in [0, 1]
        with tf.name_scope("Add_gaussian_noise"):
            noise = tf.random.normal(
                shape=tf.shape(image),
                mean=0.0,
                stddev=(self.noise_factor) / (255),
                dtype=tf.float32,
            )
            noise_img = image + noise
            noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
        return noise_img

    @tf.function
    def train_step(self, images, labels):
        one_hot_labels = keras.utils.to_categorical(labels, num_classes=self.classes)
        noise = tf.random.normal([np.shape(labels)[0], self.latent_dim])
        with tf.GradientTape() as disc_tape:
            generated_images = self.generator.model(
                [noise, one_hot_labels], training=True
            )

            real_output = self.discriminator.model(
                [self.add_gaussian_noise(images), one_hot_labels], training=True
            )
            fake_output = self.discriminator.model(
                [self.add_gaussian_noise(generated_images), one_hot_labels],
                training=True,
            )

            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.model.trainable_variables
        )

        if disc_loss > self.update_threshold:
            self.discriminator_optimizer.apply_gradients(
                zip(
                    gradients_of_discriminator,
                    self.discriminator.model.trainable_variables,
                )
            )

        with tf.GradientTape() as gen_tape:
            # sample some new noise
            noise = tf.random.normal([np.shape(labels)[0], self.latent_dim])
            generated_images = self.generator.model(
                [noise, one_hot_labels], training=True
            )
            fake_output = self.discriminator.model(
                [self.add_gaussian_noise(generated_images), one_hot_labels],
                training=True,
            )

            gen_loss = self.generator_loss(fake_output)
            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.model.trainable_variables
            )
            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.model.trainable_variables)
            )

        # Monitor loss.
        self.gen_loss_tracker.update_state(gen_loss)
        self.disc_loss_tracker.update_state(disc_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def write_to_log(self, epoch, d_loss, g_loss):
        row = [epoch, d_loss, g_loss]
        with open(self.csv_log, "a") as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(row)

    def reg_train(self, data):
        for epoch in range(self.train_steps):
            print(epoch)
            # pass the batches for an epoch
            data.shuffle(10000000)
            for real_images, labels in data:
                self.train_step(real_images, labels)
            if epoch % 100 == 0:
                self.generator.model.save(
                    "/home2/vcqf59/Classified/masters-project/self_attention/GAN_models/non_convex_last_layer_final_self_attention_gan_g_model_{resolution}".format(
                        resolution=self.resolution
                    )
                )
                self.discriminator.model.save(
                    "/home2/vcqf59/Classified/masters-project/self_attention/GAN_models/non_convex_last_layer_final_self_attention_gan_d_model_{resolution}".format(
                        resolution=self.resolution
                    )
                )
            self.write_to_log(
                epoch,
                self.disc_loss_tracker.result().numpy(),
                self.gen_loss_tracker.result().numpy(),
            )
            # self.generate_and_save_images()

    def fade_train(self, data):
        gan.step = int((self.fadein_steps - 1) * self.initial_alpha)
        for epoch in range(int((1 - self.initial_alpha) * self.fadein_steps)):
            print(epoch)
            # pass the batches for an epoch
            data.shuffle(10000000)
            for real_images, labels in data:
                self.train_step(real_images, labels)
            self.step = self.step + 1
            self.fadein()
            if epoch % 100 == 0:
                self.generator.model.save(
                    "/home2/vcqf59/Classified/masters-project/self_attention/GAN_models/non_convex_last_layer_final_self_attention_gan_g_model_{resolution}".format(
                        resolution=self.resolution
                    )
                )
                self.discriminator.model.save(
                    "/home2/vcqf59/Classified/masters-project/self_attention/GAN_models/non_convex_last_layer_final_self_attention_gan_d_model_{resolution}".format(
                        resolution=self.resolution
                    )
                )
            self.write_to_log(
                epoch,
                self.disc_loss_tracker.result().numpy(),
                self.gen_loss_tracker.result().numpy(),
            )
            # self.generate_and_save_images()
        gan.initial_alpha = 0

    def generate_and_save_images(self):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        test_input = tf.random.normal([16, self.latent_dim])
        test_labels = np.random.randint(0, self.classes, 16)
        one_hot_test_labels = keras.utils.to_categorical(
            test_labels, num_classes=self.classes
        )
        predictions = self.generator.model(
            [test_input, one_hot_test_labels], training=False
        )

        fig = plt.figure(figsize=(4, 4))
        temp = []
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            temp = np.array(predictions[i][:, :, :3])
            plt.annotate(str(test_labels[i]), xy=(1, 1))
            plt.imshow(temp)
            plt.axis("off")
        plt.show()


initial_resolution = 32
print("transfer learning from ", str(initial_resolution))
threshold = 0
noise = 0
# create log
save_file = "/home2/vcqf59/Classified/masters-project/self_attention/training_logs/non_convex_last_layer_self_att_32_75_gan.log"
csv_logger = tf.keras.callbacks.CSVLogger(save_file, append=True)

print("discriminator threshold update requirement: d_loss> ", threshold)
gan = ConditionalGAN(
    initial_resolution,
    128,
    final_resolution=75,
    csv_log=save_file,
    update_threshold=threshold,
    classes=2,
    noise_factor=noise,
    transfer=False,
)
# dont need to build because we are inserting a pre-built model
# gan.generator.model = g_model
# gan.discriminator.model = d_model
gan.build()
gan.initial_alpha = 0
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

tf.config.run_functions_eagerly(True)
resolution_increase = int(
    np.log2(gan.final_resolution / gan.resolution)
)  # how many times the resolution will double
# batch set
gan.batch_set()
# epoch set
gan.epoch_set()
data = dataset()
data.get_data(gan.resolution, gan.batch_size)
for resolution in range(resolution_increase):
    if gan.initial_alpha == 0:
        # initial_alpha is reset after finishing the first fadein train
        print("training for ", gan.train_steps, " epochs")
        gan.reg_train(data.train_dataset)

        print("growing")
        gan.grow()

        # batch set
        gan.batch_set()
        # epoch set
        gan.epoch_set()
        data.get_data(gan.resolution, gan.batch_size)

    gan.fade_train(data.train_dataset)
    print("pruning")
    gan.prune()

# add the final output layer
gan.reg_train(data.train_dataset)
# grow the output layer
gan.final_growth()
gan.batch_set()
gan.epoch_set()
data.get_data(gan.resolution, gan.batch_size)
gan.fade_train(data.train_dataset)
gan.prune()
gan.reg_train(data.train_dataset)
