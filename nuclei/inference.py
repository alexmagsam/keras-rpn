import os
import numpy as np
from lib.model import RPN
from lib import utils as ut
from nuclei.train import NucleiSequence, NucleiConfig


VALIDATION_PATH = r'A:\Deep Learning Datasets\nucleus\validation'


class NucleiInferenceConfig(NucleiConfig):

    # Data parameters
    IMAGE_SHAPE = (512, 512)
    ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ANCHORS_PER_IMAGE = 128
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    LOGS = '../logs'

    # Path to the weights file
    WEIGHTS_FILE = '../logs/nuclei/06-27-19_11.28.19/cnn_weights/rpn_weights.10.hdf5'


def main():

    # Configuration
    config = NucleiInferenceConfig()

    # Nucleus dataset
    dataset = NucleiSequence(VALIDATION_PATH, config)

    # Select a random sample from the validation set
    random_idx = np.random.randint(0, len(dataset))
    inputs, _ = dataset[random_idx]
    random_idx = np.random.randint(0, len(inputs[0]))

    # Select a random batch index
    image_gt = inputs[0][random_idx]
    rpn_match_gt = inputs[1][random_idx]
    rpn_bbox_gt = inputs[2][random_idx]

    # Create the ground truth bounding boxes
    anchors = dataset.anchors
    positive_anchors = np.where(rpn_match_gt == 1)[0]
    bboxes_gt = ut.shift_bboxes(anchors[positive_anchors], rpn_bbox_gt[:positive_anchors.shape[0]] * config.RPN_BBOX_STD_DEV)

    # Visualize the ground truth bounding boxes
    ut.visualize_bboxes(np.uint8(image_gt + config.MEAN_PIXEL), bboxes_gt)

    # Visualize the RPN targets of positive and negative anchors
    ut.visualize_training_anchors(anchors, rpn_match_gt, np.uint8(image_gt + config.MEAN_PIXEL))

    # Create the Region Proposal Network and load the trained weights
    assert os.path.exists(config.WEIGHTS_FILE)
    rpn = RPN(config, 'inference')
    rpn.model.load_weights(config.WEIGHTS_FILE, by_name=True)

    # Predict the positive anchors
    rpn_match, rpn_bbox = rpn.model.predict(np.expand_dims(image_gt, 0))
    rpn_match = np.squeeze(rpn_match)
    rpn_bbox = np.squeeze(rpn_bbox) * config.RPN_BBOX_STD_DEV

    # Visualize the predictions
    ut.visualize_rpn_predictions(np.uint8(image_gt + config.MEAN_PIXEL), rpn_match, rpn_bbox, anchors, top_n=150)


if __name__ == '__main__':
    main()
