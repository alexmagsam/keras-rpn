import numpy as np
from keras.utils import Sequence


class DataSequence(Sequence):

    def __init__(self, config, **kwargs):
        return

    def __len__(self):
        return 0

    def __getitem__(self, idx):

        inputs = []
        outputs = []

        return inputs, outputs

    def load_image(self, _id):
        """Override this method to load an image into memory using the image filename as the input

        Arguments
        ---------
        _id: str
            Filename of the image

        Returns
        -------
        Numpy array [height, width, channels]

        """
        return np.array([])

    def get_bboxes(self, _id):
        """Override this method to load the ground truth bounding boxes into memory using the image filename
            as the input

        Arguments
        ---------
        _id: str
            Filename of the image

        Returns
        -------
        Numpy array [num_bboxes, 4]

        """
        return np.array([])

    @staticmethod
    def preprocess_image(image):
        """Override this method to load an image into memory using the image filename as the input

        Arguments
        ---------
        image: Numpy array [height, width, channels]

        Returns
        -------
        preprocessed_image: Numpy array [height, width, channels]

        """
        preprocessed_image = image.astype("float32")
        return preprocessed_image
