import os
import keras.layers as KL
import keras.models as KM
import keras.optimizers as KO
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint

from lib import losses as ls


class RPN:

    def __init__(self, config, mode='train'):
        assert mode in ['train', 'inference']
        self.config = config

        # Build the model
        self.model = self.build_entire_model(mode)
        print(self.model.summary())

        # Compile in training mode
        if mode == 'train':
            self.compile()

    @staticmethod
    def build_backbone(input_tensor, architecture, stage5=False, train_bn=None):
        """Build a ResNet model.

        Arguments
        ----------
        input_tensor: Keras Input layer
            Tensor for image input
        architecture: str, "resnet50" or "resnet101"
            Architecture to use
        stage5: bool
            If False, stage5 of the network is not created
        train_bn: bool.
            Train or freeze Batch Normalization layers

        Returns
        -------
        list
            Backbone layers of ResNet 50 or 101

        """

        # Code adopted from:
        # https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

        def identity_block(tensor, kernel_size, filters, stage, block, use_bias=True):
            """The identity_block is the block that has no convolution layer at shortcut

            Arguments
            --------
            tensor: Keras Layer
                The tensor to connect to this block.
            kernel_size: int
                The kernel size of the convolutional layer
            filters: list
                List of integers indicating how many filters to use for each convolution layer
            stage: int
                Current stage label for generating layer names
            block: str
                Current block label for generating layer names
            use_bias: bool
                To use or not use a bias in conv layers.

            Returns
            -------
            y: Keras Layer
                Output of the Resnet identity block
            """

            nb_filter1, nb_filter2, nb_filter3 = filters
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            y = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(tensor)
            y = KL.BatchNormalization(name=bn_name_base + '2a')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                          use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2b')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2c')(y, training=train_bn)

            y = KL.Add()([y, tensor])
            y = KL.Activation('relu', name='res' + str(stage) + block + '_out')(y)
            return y

        def conv_block(tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True):

            """conv_block is the block that has a conv layer at shortcut

            Arguments
            ---------
            tensor: Keras Layer
                The tensor to connect to this block.
            kernel_size: int
                The kernel size of the convolutional layer
            filters: list
                List of integers indicating how many filters to use for each convolution layer
            stage: int
                Current stage label for generating layer names
            block: str
                Current block label for generating layer names
            strides: tuple
                A tuple of integers indicating the strides to make during convolution.
            use_bias: bool
                To use or not use a bias in conv layers.

            Returns
            -------
            y: Keras Layer
                Output layer of Resnet conv block

            """
            nb_filter1, nb_filter2, nb_filter3 = filters
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            y = KL.Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(
                tensor)
            y = KL.BatchNormalization(name=bn_name_base + '2a')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                          use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2b')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2c')(y, training=train_bn)

            shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(
                tensor)
            shortcut = KL.BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

            y = KL.Add()([y, shortcut])
            y = KL.Activation('relu', name='res' + str(stage) + block + '_out')(y)
            return y

        assert architecture in ["resnet50", "resnet101"]
        # Stage 1
        x = KL.ZeroPadding2D((3, 3))(input_tensor)
        x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
        x = KL.BatchNormalization(name='bn_conv1')(x, training=train_bn)
        x = KL.Activation('relu')(x)
        C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        # Stage 2
        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
        # Stage 3
        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
        # Stage 4
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
        C4 = x
        # Stage 5
        if stage5:
            x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
            C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        else:
            C5 = None
        return [C1, C2, C3, C4, C5]

    def build_feature_maps(self, input_tensor):

        """Build the feature maps for the feature pyramid.

        Arguments
        ---------
        input_tensor: Keras Input layer [height, width, channels]

        Returns
        -------
        list
            Pyramid layers

        """

        # Don't create the head (stage 5), so we pick the 4th item in the list.
        _, C2, C3, C4, C5 = self.build_backbone(input_tensor, self.config.BACKBONE, stage5=True,
                                                train_bn=self.config.TRAIN_BN)

        # Top-down Layers
        P5 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])

        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)

        # P6 is used for the 5th anchor scale in RPN. Generated by sub-sampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        return [P2, P3, P4, P5, P6]

    @staticmethod
    def build_rpn_model(anchor_stride, anchors_per_location, depth):
        """Builds a Keras model of the Region Proposal Network.

        Arguments
        ---------
        anchor_stride: int
        Controls the density of anchors. Typically 1 (anchors for every pixel in the feature map), or 2.
        anchors_per_location: int
            Number of anchors per pixel in the feature map. Equivalent to length of anchor ratios.
        depth: int,
            Depth of the backbone feature map. Same as TOP_DOWN_PYRAMID_SIZE

        Returns
        -------
        Keras Model

        The model outputs, when called, are:
            rpn_class_logits: [batch, H * W * anchors_per_location, 2]
                Anchor classifier logits (before softmax)
            rpn_probs: [batch, H * W * anchors_per_location, 2]
                Anchor classifier probabilities.
            rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))]
                Deltas to be applied to anchors.

        """

        input_feature_map = KL.Input(shape=[None, None, depth], name="input_rpn_feature_map")

        # Shared convolutional base of the RPN
        shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride,
                           name='rpn_conv_shared')(input_feature_map)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear',
                      name='rpn_class_raw')(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

        # Softmax on last dimension of BG/FG.
        rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')(
            shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

        outputs = [rpn_class_logits, rpn_probs, rpn_bbox]
        return KM.Model([input_feature_map], outputs, name="rpn_model")

    def build_entire_model(self, mode='train'):

        assert mode in ['train', 'inference']

        # Input image
        input_tensor = KL.Input(shape=[self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1],
                                       self.config.NUM_CHANNELS], name="input_image")

        # RPN feature maps
        rpn_feature_maps = self.build_feature_maps(input_tensor)

        # RPN Network
        rpn = self.build_rpn_model(self.config.ANCHOR_STRIDE, len(self.config.ANCHOR_RATIOS),
                                   self.config.TOP_DOWN_PYRAMID_SIZE)

        # Restructures [[a1, b1, c1], [a2, b2, c2]] -> [[a1, a2], [b1, b2], [c1, c2]]
        layer_outputs = []
        for layer in rpn_feature_maps:
            layer_outputs.append(rpn([layer]))
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        rpn_outputs = list(zip(*layer_outputs))
        rpn_outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(rpn_outputs, output_names)]

        # Outputs of RPN
        rpn_class_logits, rpn_class, rpn_bbox = rpn_outputs

        # Loss functions
        # GT inputs to RPN
        input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
        rpn_class_loss = KL.Lambda(lambda x: ls.rpn_match_loss(*x), name="rpn_class_loss")(
                                   [input_rpn_match, rpn_class_logits])
        rpn_bbox_loss = KL.Lambda(lambda x: ls.rpn_bbox_loss(self.config, *x), name="rpn_bbox_loss")(
                                  [input_rpn_match, input_rpn_bbox, rpn_bbox])

        # Inputs and outputs of the model
        if mode == 'train':
            inputs = [input_tensor, input_rpn_match, input_rpn_bbox]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox, rpn_class_loss, rpn_bbox_loss]
        elif mode == 'inference':
            inputs = [input_tensor]
            outputs = [rpn_class, rpn_bbox]

        # Set the model attribute
        return KM.Model(inputs, outputs, name='rpn')

    def compile(self):

        # Create the optimizer
        optimizer = KO.SGD(lr=self.config.LEARNING_RATE, momentum=self.config.LEARNING_MOMENTUM,
                           clipnorm=self.config.GRADIENT_CLIP_NORM)

        # Add Losses
        self.model._losses = []
        self.model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss"]
        for name in loss_names:
            layer = self.model.get_layer(name)
            if layer.output in self.model.losses:
                continue
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.model.add_loss(loss)

        self.model.compile(optimizer=optimizer, loss=[None] * len(self.model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.model.metrics_names:
                continue
            layer = self.model.get_layer(name)
            self.model.metrics_names.append(name)
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.model.metrics_tensors.append(loss)

    def train(self, dataset):

        """Train the region proposal network using training and validation data

        Arguments
        ---------
        dataset: dict
            Dictionary with 'train' and 'validation' keys that hold custom instances of a DataSequence in data_utils.py
            that is dataset dependent.

        """

        # Create the training directories
        self.config.create_training_directory()

        # Create a callback for saving weights
        filename = "rpn_weights.{epoch:02d}.hdf5"
        callbacks = [ModelCheckpoint(os.path.join(self.config.CNN_WEIGHTS_PATH, filename), save_weights_only=True)]

        # Create a callback for logging training information
        callbacks.append(CSVLogger(os.path.join(self.config.LOGS, self.config.NAME,
                                                self.config.TIME_STAMP, 'training.csv')))

        # Train the model
        self.model.fit_generator(dataset["train"], len(dataset["train"]), epochs=self.config.EPOCHS, callbacks=callbacks,
                                 validation_data=dataset["validation"], validation_steps=len(dataset["validation"]))

