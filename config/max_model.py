import os

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self,
                 adaption_layer_filters=[512, 512, 512],
                 adaption_kernels_size=[[3, 3], [3, 3], [3, 3]],
                 adaption_layer_strides=[(1, 1), (1, 1), (1, 1)],
                 weight_decay=1e-5,
                 plus_global_feature=True,
                 net_global_dim=[64, 128],
                 net_max_features_nums=512,
                 adaption_fc_layers_num=None,
                 adaption_fc_filters=None,
                 stages=[6],
                 annotation_number=10,
                 max_sequence_length=15,
                 annot_min_per_group=0,
                 only_word=None,
                 deprecated_word=None,
                 min_annot_num=0,
                 vgg_trainable=False,
                 vgg_output_layer='conv4/conv4_3',
                 loss_ratio=5.0,
                 neg_threshold=0.2,
                 pos_threshold=0.9
                 ):
        """Model hyperparameters and configuration."""

        # model arch parameters
        self.adaption_layer_filters = adaption_layer_filters
        self.adaption_kernels_size = adaption_kernels_size
        self.adaption_layer_strides = adaption_layer_strides
        self.adaption_fc_layers_num = adaption_fc_layers_num
        self.adaption_fc_filters = adaption_fc_filters
        self.weight_decay = weight_decay

        self.plus_global_feature = plus_global_feature
        self.net_global_dim = net_global_dim
        self.net_max_features_nums = net_max_features_nums

        # vgg parameters
        self.vgg_trainable = vgg_trainable
        self.vgg_output_layer = vgg_output_layer

        # loss parameters
        self.loss_ratio = loss_ratio
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

        # choose stage
        self.stages = stages

        # only the most k lables will be considered
        self.annotation_number = annotation_number
        
        # max image num in a group, also batch size
        self.max_sequence_length = max_sequence_length

        # min annot number in a group
        self.annot_min_per_group = annot_min_per_group

        # only the annot in only_word will be considered
        self.only_word = only_word

        # remove the specified word in deprecated_word
        self.deprecated_word = deprecated_word

        # if top_k_labels is not None, then the labels less than min_annot_num
        # will not be considered, only work when top_k_labels is None
        self.min_annot_num = min_annot_num