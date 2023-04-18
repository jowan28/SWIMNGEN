# USAGE
# python train_esrgan.py --device gpu
# python train_esrgan.py --device tpu
# import tensorflow and fix the random seed for better reproducibility
import tensorflow as tf
# import the necessary packages
#from pyimagesearch.data_preprocess import load_dataset
from ESRGAN_files.esrgan_training import ESRGANTraining
from ESRGAN_files.esrgan import ESRGAN
from ESRGAN_files.losses import Losses
from ESRGAN_files.vgg import VGG
from ESRGAN_files import config
from tensorflow import distribute
from tensorflow.config import experimental_connect_to_cluster
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow.keras.optimizers import Adam
from tensorflow.io.gfile import glob
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

# define the multi-gpu strategy
strategy = distribute.MirroredStrategy()
# set the train TFRecords, pretrained generator, and final
# generator model paths to be used for GPU training
tfrTrainPath = config.GPU_DIV2K_TFR_TRAIN_PATH
pretrainedGenPath = config.GPU_PRETRAINED_GENERATOR_MODEL
genPath = config.GPU_GENERATOR_MODEL

print(f"[INFO] number of accelerators: {strategy.num_replicas_in_sync}...")

# grab train TFRecord filenames
print("[INFO] grabbing the train TFRecords...")
trainTfr = glob(tfrTrainPath +"/*.tfrec")
# build the div2k datasets from the TFRecords
print("[INFO] creating train and test dataset...")
#trainDs = load_dataset(filenames=trainTfr, train=True,
#    batchSize=config.TRAIN_BATCH_SIZE * strategy.num_replicas_in_sync)
res = 75

#load the SWIMNSEG datasets
#load this res and its double
print('loading low res')
file_name = '/home2/vcqf59/Classified/masters-project/upscaler/np_swimnseg_'+str(res)
with open(file_name+'.pk', "rb") as output_file:
    low_res = pickle.load(output_file)
    low_res = low_res[:3200,:,:,:]
print('loading high res')
file_name = '/home2/vcqf59/Classified/masters-project/upscaler/np_swimnseg_'+str(2*res)
with open(file_name+'.pk', "rb") as output_file:
    high_res = pickle.load(output_file)
    high_res = high_res[:3200,:,:,:]
print('loaded high res')
print(np.shape(high_res))

# call the strategy scope context manager
with strategy.scope():
    # initialize our losses class object
    losses = Losses(numReplicas=strategy.num_replicas_in_sync)
    # initialize the generator, and compile it with Adam optimizer and
    # MSE loss
    generator = ESRGAN.generator(
        scalingFactor=config.SCALING_FACTOR,
        featureMaps=config.FEATURE_MAPS,
        residualBlocks=config.RESIDUAL_BLOCKS,
        leakyAlpha=config.LEAKY_ALPHA,
        residualScalar=config.RESIDUAL_SCALAR)
    generator.compile(optimizer=Adam(learning_rate=config.PRETRAIN_LR),
        loss=losses.mse_loss)
    # pretraining the generator
    print("[INFO] pretraining ESRGAN generator ...")
    generator.fit(low_res,high_res,epochs=config.PRETRAIN_EPOCHS)
    
# check whether output model directory exists, if it doesn't, then
# create it
if os.path.isdir(config.BASE_OUTPUT_PATH) == False:
    os.makedirs(config.BASE_OUTPUT_PATH)
# save the pretrained generator
print("[INFO] saving the pretrained generator...")
generator.save(pretrainedGenPath)
# call the strategy scope context manager
with strategy.scope():
    # initialize our losses class object
    losses = Losses(numReplicas=strategy.num_replicas_in_sync)
    # initialize the vgg network (for perceptual loss) and discriminator
    # network
    vgg = VGG.build()
    discriminator = ESRGAN.discriminator(
        featureMaps=config.FEATURE_MAPS,
        leakyAlpha=config.LEAKY_ALPHA,
        discBlocks=config.DISC_BLOCKS)
    # build the ESRGAN model and compile it
    esrgan = ESRGANTraining(
        generator=generator,
        discriminator=discriminator,
        vgg=vgg,
        batchSize=config.TRAIN_BATCH_SIZE,
        csv_log=config.csv_log
        )
    esrgan.compile(
        dOptimizer=Adam(learning_rate=config.FINETUNE_LR),
        gOptimizer=Adam(learning_rate=config.FINETUNE_LR),
        bceLoss=losses.bce_loss,
        mseLoss=losses.mse_loss,
    )
    # train the ESRGAN model
    print("[INFO] training ESRGAN...")
    esrgan.fit(low_res,high_res,epochs=config.FINETUNE_EPOCHS,batch_size=config.TRAIN_BATCH_SIZE)
# save the ESRGAN generator
print("[INFO] saving ESRGAN generator to {}...".format(genPath))
esrgan.generator.save(genPath)