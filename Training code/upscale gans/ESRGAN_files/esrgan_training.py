# import the necessary packages
from tensorflow.keras import Model
from tensorflow import concat
from tensorflow import zeros
from tensorflow import ones
from tensorflow import GradientTape
from tensorflow.keras.activations import sigmoid
from tensorflow.math import reduce_mean
import tensorflow as tf
import numpy as np
import csv


class ESRGANTraining(Model):
    def __init__(self, generator, discriminator, vgg, batchSize, csv_log):
        # initialize the generator, discriminator, vgg model, and
        # the global batch size
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg
        self.batchSize = batchSize
        self.csv_log = csv_log

    def compile(self, gOptimizer, dOptimizer, bceLoss, mseLoss):
        super().compile()
        # initialize the optimizers for the generator
        # and discriminator
        self.gOptimizer = gOptimizer
        self.dOptimizer = dOptimizer

        # initialize the loss functions
        self.bceLoss = bceLoss
        self.mseLoss = mseLoss

    def write_to_log(self, dLoss, gTotalLoss, gLoss, percLoss, pixelLoss):
        row = [dLoss, gTotalLoss, gLoss, percLoss, pixelLoss]
        with open(self.csv_log, "a") as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(row)

    def train_step(self, images):
        # grab the low and high resolution images
        (lrImages, hrImages) = images
        lrImages = tf.cast(lrImages, tf.float32)
        hrImages = tf.cast(hrImages, tf.float32)
        print(hrImages.get_shape())
        # generate super resolution images
        srImages = self.generator(lrImages)
        # combine them with real images
        combinedImages = concat([srImages, hrImages], axis=0)
        # assemble labels discriminating real from fake images where
        # label 0 is for predicted images and 1 is for original high
        # resolution images
        labels = concat([zeros((self.batchSize, 1)), ones((self.batchSize, 1))], axis=0)

        # train the discriminator with relativistic error
        with GradientTape() as tape:
            # get the raw predictions and divide them into
            # raw fake and raw real predictions
            rawPreds = self.discriminator(combinedImages)
            rawFake = rawPreds[: self.batchSize]
            rawReal = rawPreds[self.batchSize :]

            # process the relative raw error and pass it through the
            # sigmoid activation function
            predFake = sigmoid(rawFake - reduce_mean(rawReal))
            predReal = sigmoid(rawReal - reduce_mean(rawFake))

            # concat the predictions and calculate the discriminator
            # loss
            predictions = concat([predFake, predReal], axis=0)
            dLoss = self.bceLoss(labels, predictions)
        # compute the gradients
        grads = tape.gradient(dLoss, self.discriminator.trainable_variables)

        # optimize the discriminator weights according to the
        # gradients computed
        self.dOptimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )
        # generate misleading labels
        print(tf.shape(hrImages))
        misleadingLabels = ones((self.batchSize, 1))

        # train the generator (note that we should *not* update
        # the weights of the discriminator)
        with GradientTape() as tape:
            # generate fake images
            fakeImages = self.generator(lrImages)

            # calculate predictions
            rawPreds = self.discriminator(fakeImages)
            realPreds = self.discriminator(hrImages)
            relativisticPreds = rawPreds - reduce_mean(realPreds)
            predictions = sigmoid(relativisticPreds)

            # compute the discriminator predictions on the fake images
            # todo: try with logits
            # gLoss = self.bceLoss(misleadingLabels, predictions)
            print("misleading labels", misleadingLabels, "predictions", predictions)
            gLoss = self.bceLoss(misleadingLabels, predictions)

            # compute the pixel loss
            pixelLoss = self.mseLoss(hrImages, fakeImages)
            fakeVGG_im = fakeImages[:, :, :, :3]
            realVGG_im = hrImages[:, :, :, :3]
            # compute the normalized vgg outputs
            srVGG = tf.keras.applications.vgg19.preprocess_input(fakeVGG_im)
            srVGG = self.vgg(srVGG) / 12.75
            hrVGG = tf.keras.applications.vgg19.preprocess_input(realVGG_im)
            hrVGG = self.vgg(hrVGG) / 12.75
            # compute the perceptual loss
            percLoss = self.mseLoss(hrVGG, srVGG)

            # compute the total GAN loss
            gTotalLoss = 5e-3 * gLoss + percLoss + 1e-2 * pixelLoss

        # compute the gradients
        grads = tape.gradient(gTotalLoss, self.generator.trainable_variables)

        # optimize the generator weights according to the gradients
        # calculated
        self.gOptimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        # return the generator and discriminator losses
        self.write_to_log(dLoss, gTotalLoss, gLoss, percLoss, pixelLoss)
        return {
            "dLoss": dLoss,
            "gTotalLoss": gTotalLoss,
            "gLoss": gLoss,
            "percLoss": percLoss,
            "pixelLoss": pixelLoss,
        }
