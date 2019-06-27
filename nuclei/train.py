import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from lib.config import Config
from lib.data_utils import DataSequence
from lib.model import RPN
from lib import utils as ut


TRAIN_PATH = r'nucleus\train'
VALIDATION_PATH = r'nucleus\validation'


class NucleiConfig(Config):

    NAME = 'nuclei'

    # Data parameters
    IMAGE_SHAPE = (512, 512)
    ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ANCHORS_PER_IMAGE = 128
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # Learning parameters
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    BATCH_SIZE = 8
    EPOCHS = 10


class NucleiSequence(DataSequence):

    def __init__(self, path, config):
        super().__init__(config)

        # Get the path to the data
        self.path = path

        # Image IDs are the folder names in this dataset
        self.image_ids = next(os.walk(self.path))[1]
        np.random.shuffle(self.image_ids)

        # Store the configuration class
        self.config = config

        # Generate the anchors
        self.anchors = ut.generate_anchors(self.config.ANCHOR_SCALES,
                                           self.config.ANCHOR_RATIOS,
                                           ut.backbone_shapes(self.config.IMAGE_SHAPE, self.config.BACKBONE_STRIDES),
                                           self.config.BACKBONE_STRIDES,
                                           self.config.ANCHOR_STRIDE)

    def __len__(self):
        return int(len(self.image_ids) / self.config.BATCH_SIZE)

    def __getitem__(self, idx):

        # Choose the image ID's to be loaded into the batch
        image_ids = self.image_ids[idx * self.config.BATCH_SIZE: (idx + 1) * self.config.BATCH_SIZE]

        # Only RGB images - todo: fix this
        image_batch = np.zeros(((self.config.BATCH_SIZE, ) + self.config.IMAGE_SHAPE + (3, )))
        rpn_match_batch = np.zeros((self.config.BATCH_SIZE, self.anchors.shape[0], 1))
        rpn_bbox_batch = np.zeros((self.config.BATCH_SIZE, self.config.TRAIN_ANCHORS_PER_IMAGE, 4))

        # Load the batches
        for batch_idx in range(self.config.BATCH_SIZE):

            # Load the image and
            image = self.load_image(image_ids[batch_idx])
            bboxes = self.get_bboxes(image_ids[batch_idx])

            # Trim bboxes
            if bboxes.shape[0] > self.config.MAX_GT_INSTANCES:
                bboxes = bboxes[:self.config.MAX_GT_INSTANCES]

            # Generate the ground truth RPN targets to learn from
            rpn_match, rpn_bbox = ut.rpn_targets(self.anchors, bboxes, self.config)

            # Update the batch variables
            image_batch[batch_idx] = self.preprocess_image(image)
            rpn_match_batch[batch_idx] = np.expand_dims(rpn_match, axis=1)
            rpn_bbox_batch[batch_idx] = rpn_bbox

        # Store the inputs in a list form
        inputs = [image_batch, rpn_match_batch, rpn_bbox_batch]

        return inputs, []

    def load_image(self, _id):
        filename = os.path.join(self.path, _id, 'images', _id + '.png')
        return img_to_array(load_img(filename, target_size=self.config.IMAGE_SHAPE))

    def get_bboxes(self, _id):

        # Get the filenames for all of the nuclei masks
        filenames = next(os.walk(os.path.join(self.path, _id, 'masks')))[2]
        bboxes = []

        # Loop through each nuclei mask
        for idx, name in enumerate(filenames):

            # Load the nuclei mask
            filename = os.path.join(self.path, _id, 'masks', name)
            mask = img_to_array(load_img(filename, color_mode='grayscale', target_size=self.config.IMAGE_SHAPE))

            # Find the positive indices and create the bounding box
            positive_idxs = np.transpose(np.where(mask > (255 / 2)))
            if np.any(positive_idxs):
                bboxes.append([np.min(positive_idxs[:, 0]), np.min(positive_idxs[:, 1]),
                               np.max(positive_idxs[:, 0]), np.max(positive_idxs[:, 1])])

        return np.array(bboxes)

    def preprocess_image(self, image):
        # Subtract the mean
        preprocessed_image = image.astype("float32") - self.config.MEAN_PIXEL
        return preprocessed_image


def main():

    # Configuration
    config = NucleiConfig()

    # Dataset
    dataset = {"train": NucleiSequence(TRAIN_PATH, config), "validation": NucleiSequence(VALIDATION_PATH, config)}

    # Region proposal network
    rpn = RPN(config)
    rpn.train(dataset)


if __name__ == '__main__':
    main()