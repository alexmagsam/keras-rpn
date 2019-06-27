import keras.engine as KE
import keras.backend as K
import tensorflow as tf
import lib.graph_utils as gu


class Proposal(KE.Layer):

    def __init__(self, pre_nms_limit, post_nms_limit, nms_threshold, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.pre_nms_limit = pre_nms_limit
        self.post_nms_limit = post_nms_limit
        self.nms_threshold = nms_threshold

    def call(self, inputs, **kwargs):

        # Give the inputs names
        rpn_match = inputs[0]
        rpn_bbox = inputs[1]
        anchors = inputs[2]

        # Loop through batches
        proposals_batch = []
        for batch_idx in range(self.batch_size):

            # Get top pre_nms_limit anchor indices
            top_k = tf.nn.top_k(rpn_match[batch_idx, :, 1], self.pre_nms_limit, sorted=True, name="top_anchors")

            # Get top pre_nms_limit scores
            top_scores = K.gather(rpn_match[batch_idx, :, 1], top_k.indices)

            # Get top pre_nms_limit anchor shifts
            top_shifts = K.gather(rpn_bbox[batch_idx], top_k.indices)

            # Get top pre_nms_limit anchors
            top_anchors = K.gather(anchors[batch_idx], top_k.indices)

            # Shift the top anchors
            shifted_anchors = gu.shift_anchors(top_anchors, top_shifts)

            # Get indices of proposals after non-max suppression
            proposal_idxs = tf.image.non_max_suppression(shifted_anchors, top_scores, self.post_nms_limit,
                                                         self.nms_threshold)

            # Pad if necessary
            padding = K.maximum(self.post_nms_limit - K.shape(proposal_idxs)[0], 0)
            proposals = tf.pad(K.gather(shifted_anchors, proposal_idxs), [(0, padding), (0, 0)])

            # Append to list
            proposals_batch.append(proposals)

        return K.stack(proposals_batch, axis=0)

    def compute_output_shape(self, input_shape):
        return (None, self.post_nms_limit, 4)