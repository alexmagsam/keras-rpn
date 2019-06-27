import keras.backend as K
import tensorflow as tf
from lib import graph_utils as gu


def smooth_l1_loss(y_true, y_pred):
    """Calculates Smooth L1 loss

    Parameters
    ----------
    y_true: [None, 4]
        Ground truth bounding box shifts
    y_pred: [None, 4]
        Predicted bounding box shifts

    Returns
    -------
    Smooth L1 loss
    """

    # Take absolute difference
    x = K.abs(y_true - y_pred)

    # Find indices of values less than 1
    mask = K.cast(K.less(x, 1.0), "float32")

    # Loss calculation for smooth l1
    loss = (mask * (0.5 * x ** 2)) + (1 - mask) * (x - 0.5)
    return loss


def rpn_match_loss(rpn_match_gt, rpn_match_logits):
    """Loss function for the RPN match output

    Parameters
    ----------
    rpn_match_gt: [batch, anchors, 1].
        Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    rpn_match_logits: [batch, anchors, 2].
        RPN classifier logits for FG/BG.

    Returns
    -------
    Cross-entropy loss for the predicted RPN match

    """

    # Squeeze last dimension
    rpn_match_gt = tf.squeeze(rpn_match_gt, -1)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match_gt, 1), tf.int32)

    # Find indices of positive and negative anchors, not neutral
    indices = tf.where(K.not_equal(rpn_match_gt, 0))

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_match_logits = tf.gather_nd(rpn_match_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)

    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_match_logits,
                                             from_logits=True)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss(config, rpn_match_gt, rpn_bbox_gt, rpn_bbox):
    """Loss function for the RPN bbox output

    Parameters
    ----------
    config: Config object
        Contains batch size used in training.
    rpn_match_gt: [batch, anchors, 1]
        Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    rpn_bbox_gt: [batch, max positive anchors, (dy, dx, log(dh), log(dw))]
        Ground truth bbox shifts.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        Predicted bbox shifts.

    Returns
    -------
    Smooth-L1 loss
    """

    loss = []
    for batch_idx in range(config.BATCH_SIZE):

        # Find indices for positive anchors
        match = K.squeeze(rpn_match_gt[batch_idx], -1)
        positive_idxs = tf.where(K.equal(match, 1))

        # Select positive predicted bbox shifts
        bbox = tf.gather_nd(rpn_bbox[batch_idx], positive_idxs)

        # Trim target bounding box deltas to the same length as rpn_bbox
        target_bbox = rpn_bbox_gt[batch_idx, :K.shape(positive_idxs)[0]]

        # Calculate the loss for the batch
        loss.append(smooth_l1_loss(target_bbox, bbox))

    return K.mean(K.concatenate(loss, 0))


