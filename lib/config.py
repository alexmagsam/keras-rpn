import os
import datetime
import numpy as np


class Config(object):

    # Name of the experiment
    NAME = None

    # Default logs directory to save weights of experiments
    LOGS = 'logs'

    # Time stamp of the experiment - generated automatically
    TIME_STAMP = None

    # Path to save trained weights in - generated automatically
    CNN_WEIGHTS_PATH = None

    # Batch size
    BATCH_SIZE = 2

    # Backbone
    BACKBONE = "resnet50"

    # Backbone strides to make feature map shapes
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Length of square anchor side in pixels
    ANCHOR_SCALES = (16, 32, 64, 128)

    # Ratios of anchors at each cell (width/height)
    ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    ANCHOR_STRIDE = 1

    # How many anchors per image to use for RPN training
    TRAIN_ANCHORS_PER_IMAGE = 64

    # Max ground truth bounding boxes
    MAX_GT_INSTANCES = 100

    # Input image resizing
    IMAGE_SHAPE = (512, 512)

    # Number of channels
    NUM_CHANNELS = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Learning rate and momentum
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    LOSS_WEIGHTS = {"rpn_class_loss": 1., "rpn_bbox_loss": 1.}

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        assert os.path.exists(self.LOGS)

    def create_training_directory(self):

        self.TIME_STAMP = datetime.datetime.strftime(datetime.datetime.now(), "%m-%d-%y_%H.%M.%S")

        try:
            os.mkdir(os.path.join(self.LOGS, self.NAME))
        except FileExistsError:
            pass

        try:
            os.mkdir(os.path.join(self.LOGS, self.NAME, self.TIME_STAMP))
        except FileExistsError:
            pass

        self.CNN_WEIGHTS_PATH = os.path.join(self.LOGS, self.NAME, self.TIME_STAMP, "cnn_weights")
        try:
            os.mkdir(self.CNN_WEIGHTS_PATH)
        except FileExistsError:
            pass