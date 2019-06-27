import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def backbone_shapes(image_shape, backbone_strides):
    """Generates shapes of feature maps

    Parameters
    ----------
    image_shape: [height, width]
        Shape of the input image
    backbone_strides: list, len()=5
        Downscaling factors based on strides used in creation of the feature maps.

    Returns
    -------
    list of tuples (height, width) of feature map shapes

    """
    return [(image_shape[0] // stride, image_shape[1] // stride) for stride in backbone_strides]


def generate_anchors(anchor_sizes, anchor_ratios, feature_shapes, feature_strides, anchor_stride):
    """Generates all potential anchors in an image.

    Parameters
    ----------
    anchor_sizes: list
        Different sizes of anchors to use.
    anchor_ratios: list
        Ratio of width/height for the anchors.
    feature_shapes: list
        List of the backbone feature map shapes.
    feature_strides: list
        Downscaling factors used to calculate feature map shapes.
    anchor_stride: int
        Stride to use when creating anchors combinations. Typically 1.

    Returns
    -------
    anchors: [N * len(anchor_size), (y1, x1, y2, x2)]

    """

    anchors = []
    for idx in range(len(anchor_sizes)):

        # All combinations of sizes and ratios
        sizes, ratios = np.meshgrid(anchor_sizes[idx], anchor_ratios)

        # All combination of height and width
        height = sizes.flatten() / np.sqrt(ratios.flatten())
        width = sizes.flatten() * np.sqrt(ratios.flatten())

        # All combinations of indices
        x = np.arange(0, feature_shapes[idx][1], anchor_stride) * feature_strides[idx]
        y = np.arange(0, feature_shapes[idx][0], anchor_stride) * feature_strides[idx]
        x, y = np.meshgrid(x, y)

        # All combinations of indices, and shapes
        width, x = np.meshgrid(width, x)
        height, y = np.meshgrid(height, y)

        # Reshape indices and shapes
        x = x.flatten().reshape((-1, 1))
        y = y.flatten().reshape((-1, 1))
        width = width.flatten().reshape((-1, 1))
        height = height.flatten().reshape((-1, 1))

        # Create the centers coordinates and shapes for the anchors
        bbox_centers = np.concatenate((y, x), axis=1)
        bbox_shapes = np.concatenate((height, width), axis=1)

        # Restructure as [y1, x1, y2, x2]
        bboxes = np.concatenate((bbox_centers - bbox_shapes / 2, bbox_centers + bbox_shapes / 2), axis=1)

        # Anchors are created for each feature map
        anchors.append(bboxes)

    return np.concatenate(anchors, axis=0)


def rpn_targets(anchors, bbox_gt, config):
    """Build the targets for training the RPN

    Arguments
    ---------
    anchors: [N, 4]
        All potential anchors in the image
    bbox_gt: [M, 4]
        Ground truth bounding boxes
    config: Config
        Instance of the Config class that stores the parameters

    Returns
    -------
    rpn_match: [N, 1]
        Array same length as anchors with 1=positive, 0=neutral, -1=negative
    rpn_bbox: [config.TRAIN_ANCHORS_PER_IMAGE, 4]
        Array that stores the bounding box shifts needed to adjust the positive anchors by

    """

    # Outputs
    rpn_match = np.zeros(anchors.shape[0], np.float32)
    rpn_bbox = np.zeros((config.TRAIN_ANCHORS_PER_IMAGE, 4), np.float32)

    # Find the iou between all anchors and all bboxes
    iou_mat = iou_matrix(anchors, bbox_gt)

    # Find best bbox index for each anchor
    best_bboxs = np.argmax(iou_mat, axis=1)

    # Find best anchor for every bbox
    best_anchors = np.argmax(iou_mat, axis=0)

    # Create the IOU matrix
    anchor_iou = iou_mat[np.arange(0, iou_mat.shape[0]), best_bboxs]

    # Set the ground truth values for RPN match
    rpn_match[anchor_iou < .3] = -1
    rpn_match[anchor_iou > .7] = 1

    # Assign a value to all bboxes - note there will be duplicates
    rpn_match[best_anchors] = 1

    # There can only be 1:1 ratio of positive anchors to negative anchors at max
    positive_anchors = np.where(rpn_match == 1)[0]
    if len(positive_anchors) > config.TRAIN_ANCHORS_PER_IMAGE // 2:
        set_to_zero = np.random.choice(positive_anchors, len(positive_anchors) - config.TRAIN_ANCHORS_PER_IMAGE // 2,
                                       replace=False)
        # Set extras to zero
        rpn_match[set_to_zero] = 0

        # Reset positive anchors
        positive_anchors = np.where(rpn_match == 1)[0]

    # Set negative anchors to the difference between allowed number of total anchors and the positive anchors
    negative_anchors = np.where(rpn_match == -1)[0]
    set_to_zero = np.random.choice(negative_anchors,
                                   len(negative_anchors) - (config.TRAIN_ANCHORS_PER_IMAGE - len(positive_anchors)),
                                   replace=False)
    rpn_match[set_to_zero] = 0

    # Reset negative anchors
    negative_anchors = np.where(rpn_match == -1)[0]

    # Create the RPN bbox targets
    target_anchors = anchors[positive_anchors]

    # The anchor adjustments are assigned to the top half or less of rpn_bbox, the rest are zeros.
    for idx in range(target_anchors.shape[0]):

        # Get the closest bbox and target corresponding anchor
        bbox = bbox_gt[best_bboxs[positive_anchors[idx]]]
        anchor = target_anchors[idx]

        # Bbox dimensions and centroids
        bbox_height = bbox[2] - bbox[0]
        bbox_width = bbox[3] - bbox[1]
        bbox_y = np.mean([bbox[2], bbox[0]])
        bbox_x = np.mean([bbox[3], bbox[1]])

        # Anchor dimensions and centroids
        anchor_height = anchor[2] - anchor[0]
        anchor_width = anchor[3] - anchor[1]
        anchor_y = np.mean([anchor[2], anchor[0]])
        anchor_x = np.mean([anchor[3], anchor[1]])

        # Adjustment in normalized coordinates
        adjustment = np.array([(bbox_y - anchor_y) / anchor_height,
                               (bbox_x - anchor_x) / anchor_width,
                               np.log(bbox_height / anchor_height),
                               np.log(bbox_width / anchor_width)])

        # Normalize further by dividing by std
        normalized_adjustment = adjustment / config.RPN_BBOX_STD_DEV

        # Set the ground truth rpn bbox
        rpn_bbox[idx] = normalized_adjustment

    return rpn_match, rpn_bbox


def iou_matrix(anchors, bboxes):
    """Creates a matrix of IOU values for each combination of an anchor and bounding box

    Arguments
    ---------
    anchors: [N, 4]
        Array of anchors [y1, x1, y2, x2]
    bboxes: [M, 4]
        Array of bounding boxes [y1, x1, y2, x2]

    Returns
    -------
    iou: [N, M]
        Matrix of IOU values.

    """

    # All combinations of the y1, x1, y2, x2
    y1_bbox, y1_anchor = np.meshgrid(bboxes[:, 0], anchors[:, 0])
    x1_bbox, x1_anchor = np.meshgrid(bboxes[:, 1], anchors[:, 1])
    y2_bbox, y2_anchor = np.meshgrid(bboxes[:, 2], anchors[:, 2])
    x2_bbox, x2_anchor = np.meshgrid(bboxes[:, 3], anchors[:, 3])

    # Coordinates of the intersecting boxes
    y1 = np.maximum(y1_bbox, y1_anchor)
    x1 = np.maximum(x1_bbox, x1_anchor)
    y2 = np.minimum(y2_bbox, y2_anchor)
    x2 = np.minimum(x2_bbox, x2_anchor)

    # Area of the intersection boxes
    intersection = np.maximum(0, y2 - y1) * np.maximum(0, x2 - x1)

    # Area
    anchor_area = (y2_anchor - y1_anchor) * (x2_anchor - x1_anchor)
    bbox_area = (y2_bbox - y1_bbox) * (x2_bbox - x1_bbox)

    # Union
    union = anchor_area + bbox_area - intersection

    return intersection / union


def shift_bboxes(bboxes, shifts):
    """Transforms anchors based on predicted shifts.

    Arguments
    ---------
    bboxes: [N, 4]
        Bounding boxes [y1, x1, y2, x2]
    shifts: [N, 4]
        Bounding box shifts [dy/h, dx/w, log(by/ay), log(bx/ax)]

    Returns
    -------
    shifted bounding boxes [y1, x1, y2, x2]

    """

    height = bboxes[:, 2] - bboxes[:, 0]
    width = bboxes[:, 3] - bboxes[:, 1]

    center_y = bboxes[:, 0] + height / 2
    center_x = bboxes[:, 1] + width / 2

    center_y = shifts[:, 0] * height + center_y
    center_x = shifts[:, 1] * width + center_x

    height = np.exp(shifts[:, 2]) * height
    width = np.exp(shifts[:, 3]) * width

    y1 = center_y - height / 2
    y2 = center_y + height / 2

    x1 = center_x - width / 2
    x2 = center_x + width / 2

    return np.stack([y1, x1, y2, x2], axis=1)


def visualize_rpn_predictions(image, rpn_match, rpn_bbox, anchors, top_n=100):

    """Visualize RPN predictions.

    Arguments
    ---------
    image: [height, widht, channels]
        Image.
    rpn_match: [N, 2]
        Array indicating if an anchor belongs to foreground or background.
    rpn_bbox: [N, 4]
        Bounding box shifts [dy/h, dx/w, log(by/ay), log(bx/ax)]
    anchors: [N, 4]
        Anchors present in the image [y1, x1, y2, x2]
    top_n: int
        Value to indicate how many anchors to visualize

    """

    # Find where positive predictions took place
    positive_idxs = np.where(np.argmax(rpn_match, axis=1) == 1)[0]

    # Get the predicted anchors for the positive anchors
    predicted_anchors = shift_bboxes(anchors[positive_idxs], rpn_bbox[positive_idxs])

    # Sort predicted class by strength of prediction
    argsort = np.flip(np.argsort(rpn_match[positive_idxs, 1]), axis=0)
    sorted_anchors = predicted_anchors[argsort]
    sorted_anchors = sorted_anchors[:min(top_n, sorted_anchors.shape[0])]

    # One subplot
    fig, axes = plt.subplots(ncols=1)
    axes.imshow(image)
    axes.set_title("Top {} Region Proposal Network predictions".format(top_n))

    # Loop through predictions
    for a in sorted_anchors:
        rect = patches.Rectangle((a[1], a[0]), a[3] - a[1], a[2] - a[0], linewidth=1, edgecolor='r', facecolor='none',
                                 linestyle=':')
        axes.add_patch(rect)

    plt.show()


def visualize_training_anchors(anchors, rpn_match, image):
    """Visualize positive and negative anchors using for training the RPN.

    Arguments
    ---------
    anchors: [N, 4]
        Array of bounding boxes [y1, x1, y2, x2]
    rpn_match: [N, 1]
        Array indicating if an anchor at a given index is positive (=1) or negative (=-1)
    image: [height, width, channels]
        Image.

    """

    # Positve and negative anchors
    positive_anchors = anchors[np.where(rpn_match == 1)[0]]
    negative_anchors = anchors[np.where(rpn_match == -1)[0]]

    # Visualize two subplots
    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Positive Anchors : {}".format(len(positive_anchors)))
    axes[1].imshow(image, cmap='gray')
    axes[1].set_title("Negative Anchors : {}".format(len(negative_anchors)))

    # Positive anchors
    for a in positive_anchors:
        rect = patches.Rectangle((a[1], a[0]), a[3]-a[1], a[2]-a[0], linewidth=1, edgecolor='b', facecolor='none', linestyle=':')
        axes[0].add_patch(rect)

    # Negative anchors
    for a in negative_anchors:
        rect = patches.Rectangle((a[1], a[0]), a[3]-a[1], a[2]-a[0], linewidth=1, edgecolor='r', facecolor='none', linestyle=':')
        axes[1].add_patch(rect)

    plt.show()


def visualize_bboxes(image, bboxes):

    """Visualize bounding boxes in an image.

    Arguments
    ---------
    image: [height, width, channels]
        Image to be displayed.
    bboxes: [N, 4]
        Array of bounding boxes [y1, x1, y2, x2]

    """

    # One subplot
    fig, axes = plt.subplots(ncols=1)
    axes.imshow(image)
    axes.set_title("Bounding Boxes")

    # Loop through all bounding boxes
    for a in bboxes:
        rect = patches.Rectangle((a[1], a[0]), a[3]-a[1], a[2]-a[0], linewidth=2, edgecolor='r', facecolor='none', linestyle='-')
        axes.add_patch(rect)

    plt.show()
