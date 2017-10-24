from pathlib import Path


class Config():
    pass


default = Config()
default.image_directory = str(Path.home()) + "/Documents/flyexpress/DL_biomedicine_image/data/pic_data/"
default.image_size = (320, 128)
default.image_table_location = str(Path.home()) + "/Documents/flyexpress/DL_biomedicine_image/data/standard_images.csv"
default.annotation_table_location = str(Path.home()) + "/Documents/flyexpress/DL_biomedicine_image/data/standard_annotations.csv"
default.stages = [3, 4, 5, 6]
default.directions = ['ventral', 'dorsal', 'lateral']
default.annotation_number = 20
default.max_sequence_length = 10
default.proportion = {'train': 0.6, 'val': 0.2, 'test': 0.2}
default.tolerance_margin = 0.02
default.shuffle_separation = True
default.validation_metrics = ['mean_average_precision', 'macro_auc', 'micro_auc',
                              'macro_f1', 'micro_f1', 'ranking_mean_average_precision',
                              'coverage', 'ranking_loss', 'one_error']
default.use_glimpse = True
default.read_time = 5
default.weight_decay = 0.0
default.dropout_keep_prob = 0.5
default.batch_size = 64