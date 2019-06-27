import keras.backend as K
import tensorflow as tf


def target_shift(proposals, bboxes_gt):
    """Calculates shift needed to transform proposals to match bboxes_gt"""

    proposals = K.cast(proposals, 'float32')
    bboxes_gt = K.cast(bboxes_gt, 'float32')

    proposals_height = proposals[:, 2] - proposals[:, 0]
    proposals_width = proposals[:, 3] - proposals[:, 1]
    proposals_y = proposals[:, 0] + proposals_height * 0.5
    proposals_x = proposals[:, 1] + proposals_width * 0.5

    bboxes_gt_height = bboxes_gt[:, 2] - bboxes_gt[:, 0]
    bboxes_gt_width = bboxes_gt[:, 3] - bboxes_gt[:, 1]
    bboxes_gt_y = bboxes_gt[:, 0] + bboxes_gt_height * 0.5
    bboxes_gt_x = bboxes_gt[:, 1] + bboxes_gt_width * 0.5

    shift = K.stack([(bboxes_gt_y - proposals_y) / proposals_height,
                     (bboxes_gt_x - proposals_x) / proposals_width,
                     K.log(bboxes_gt_height / proposals_height),
                     K.log(bboxes_gt_width / proposals_width)], axis=1)

    return shift


def shift_anchors(anchors, shifts):
    """Shift anchors"""

    anchors = K.cast(anchors, 'float32')
    shifts = K.cast(shifts, 'float32')

    height = anchors[:, 2] - anchors[:, 0]
    width = anchors[:, 3] - anchors[:, 1]

    center_y = anchors[:, 0] + height / 2
    center_x = anchors[:, 1] + width / 2

    center_y = shifts[:, 0] * height + center_y
    center_x = shifts[:, 1] * width + center_x

    height = K.exp(shifts[:, 2]) * height
    width = K.exp(shifts[:, 3]) * width

    y1 = center_y - height / 2
    y2 = center_y + height / 2

    x1 = center_x - width / 2
    x2 = center_x + width / 2

    return K.stack([y1, x1, y2, x2], axis=1)


def iou_matrix(bboxes_1, bboxes_2):
    """Creates a IOU matrix """

    # Create grid pairs for every combination
    y1_bboxes_1, y1_bboxes_2 = tf.meshgrid(bboxes_1[:, 0], bboxes_2[:, 0])
    x1_bboxes_1, x1_bboxes_2 = tf.meshgrid(bboxes_1[:, 1], bboxes_2[:, 1])
    y2_bboxes_1, y2_bboxes_2 = tf.meshgrid(bboxes_1[:, 2], bboxes_2[:, 2])
    x2_bboxes_1, x2_bboxes_2 = tf.meshgrid(bboxes_1[:, 3], bboxes_2[:, 3])

    # Intersecting coordinates
    y1 = tf.maximum(y1_bboxes_1, y1_bboxes_2)
    x1 = tf.maximum(x1_bboxes_1, x1_bboxes_2)
    y2 = tf.minimum(y2_bboxes_1, y2_bboxes_2)
    x2 = tf.minimum(x2_bboxes_1, x2_bboxes_2)

    # Area of bboxes_1 and bboxes_2
    bboxes_1_area = (y2_bboxes_1 - y1_bboxes_1) * (x2_bboxes_1 - x1_bboxes_1)
    bboxes_2_area = (y2_bboxes_2 - y1_bboxes_2) * (x2_bboxes_2 - x1_bboxes_2)

    # Intersection and union
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    union = bboxes_1_area + bboxes_2_area - intersection

    return intersection / union


def bbox_iou(bbox, bboxes):

    # Coordinates of intersecting box
    y1 = K.maximum(bbox[0], bboxes[:, 0])
    y2 = K.minimum(bbox[2], bboxes[:, 2])
    x1 = K.maximum(bbox[1], bboxes[:, 1])
    x2 = K.minimum(bbox[3], bboxes[:, 3])

    # Area of intersection
    height = K.maximum(y2 - y1, 0)
    width = K.maximum(x2 - x1, 0)
    intersection = height * width

    # Area of union
    anchor_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    union = (anchor_area + bbox_area) - intersection

    return intersection / union


if __name__ == '__main__':

    bbox_gt = K.variable([[0, 0, 10, 10], [10, 10, 20, 40], [20, 40, 40, 50]], dtype='float32')
    prop = K.variable([[21, 39, 38, 50],
                       [1, 2, 10, 10],
                       [19, 40, 42, 49],
                       [9, 8, 19, 40],
                       [0, 2, 8, 10],
                       [10, 10, 18, 42]],
                      dtype='float32')

    iou_mat = K.transpose(iou_matrix(prop, bbox_gt))

    max_iou = K.max(iou_mat, axis=1)

    positive_idxs = tf.where(max_iou >= 0.5)[:, 0]
    negative_idxs = tf.where(max_iou < 0.5)[:, 0]

    argmax = K.argmax(K.gather(iou_mat, positive_idxs), axis=1)

    positive_bboxes_gt = K.gather(bbox_gt, argmax)

    ts = target_shift(prop, positive_bboxes_gt)

    print("Proposals")
    print(prop.eval(session=K.get_session()))

    print("\nBboxes")
    print(positive_bboxes_gt.eval(session=K.get_session()))

    print("\nShifts")
    print(ts.eval(session=K.get_session()))

    print("\nTransformed Shifts")
    print(shift_anchors(prop, ts).eval(session=K.get_session()))