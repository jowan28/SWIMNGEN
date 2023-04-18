# import the necessary packages
import os
# name of the TFDS dataset we will be using
DATASET = "div2k/bicubic_x4"
# define the shard size and batch size
SHARD_SIZE = 256
TRAIN_BATCH_SIZE = 16
INFER_BATCH_SIZE = 16
# dataset specs
HR_SHAPE = [150, 150, 4]
LR_SHAPE = [75, 75, 4]
SCALING_FACTOR = 2

# GAN model specs
FEATURE_MAPS = 64
RESIDUAL_BLOCKS = 20
LEAKY_ALPHA = 0.2
DISC_BLOCKS = 4
RESIDUAL_SCALAR = 0.2
# training specs
PRETRAIN_LR = 1e-4
FINETUNE_LR = 3e-5
PRETRAIN_EPOCHS = 1500
FINETUNE_EPOCHS = 1000
STEPS_PER_EPOCH = 10
# define the path to the dataset
BASE_DATA_PATH = "dataset"
DIV2K_PATH = os.path.join(BASE_DATA_PATH, "div2k")
csv_log = '/home2/vcqf59/Classified/masters-project/upscaler/ESRGAN_folder/final_2_times'
post_train_csv_log = '/home2/vcqf59/Classified/masters-project/upscaler/ESRGAN_folder/post_final_2_times'

# define the path to the tfrecords for GPU training
GPU_BASE_TFR_PATH = "tfrecord"
GPU_DIV2K_TFR_TRAIN_PATH = os.path.join(GPU_BASE_TFR_PATH, "train")
GPU_DIV2K_TFR_TEST_PATH = os.path.join(GPU_BASE_TFR_PATH, "test")
# define the path to the tfrecords for TPU training
TPU_BASE_TFR_PATH = "gs://<PATH_TO_GCS_BUCKET>/tfrecord"
TPU_DIV2K_TFR_TRAIN_PATH = os.path.join(TPU_BASE_TFR_PATH, "train")
TPU_DIV2K_TFR_TEST_PATH = os.path.join(TPU_BASE_TFR_PATH, "test")
# path to our base output directory
BASE_OUTPUT_PATH = "final_outputs"+str(LR_SHAPE[0])+'_to_'+str(HR_SHAPE[0])
# GPU training ESRGAN model paths
GPU_PRETRAINED_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH,
    "models", "big_attention_pretrained_generator")
GPU_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH, "models",
    "big_attention_generator")
# TPU training ESRGAN model paths
TPU_OUTPUT_PATH = "gs://<PATH_TO_GCS_BUCKET>/outputs"
TPU_PRETRAINED_GENERATOR_MODEL = os.path.join(TPU_OUTPUT_PATH,
    "models", "pretrained_generator")
TPU_GENERATOR_MODEL = os.path.join(TPU_OUTPUT_PATH, "models",
    "generator")
# define the path to the inferred images and to the grid image
BASE_IMAGE_PATH = os.path.join(BASE_OUTPUT_PATH, "images")
GRID_IMAGE_PATH = os.path.join(BASE_OUTPUT_PATH, "grid.png")