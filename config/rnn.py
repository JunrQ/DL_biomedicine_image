from collections import namedtuple


class Config(namedtuple('Config', [
        'image_directory',
        'image_size',
        'image_table_location',
        'annotation_table_location',
        'stages',
        'directions',
        'annotation_number',
        'max_sequence_length',
        'proportion',
        'tolerance_margin',
        'shuffle_separation',
        'read_time',
        'weight_decay',
        'batch_size',
])):
    pass


default = Config()
default.image_directory = "$HOME/Documents/flyexpress/DL_biomedicine_image/data/pic_data/"
default.image_size = (128, 320)
default.image_table_location = "$HOME/Documents/flyexpress/DL_biomedicine_image/data/standard_images.csv"
default.annotation_table_location = "$HOME/Documents/flyexpress/DL_biomedicine_image/data/standard_annotations.csv"
default.stages = [3, 4, 5, 6]
default.directions = ['ventral', 'dorsal', 'lateral']
default.annotation_number = 20
default.max_sequence_length = 10
default.proportion = {'train': 0.6, 'val': 0.2, 'test': 0.2}
default.tolerance_margin = 0.02
default.shuffle_separation = True
default.read_time = 5
default.weight_decay = 5e-4
default.batch_size = 32
