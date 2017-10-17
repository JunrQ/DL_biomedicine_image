import os

GRAND_PARENT_PATH = r'/home/litiange/prp_file'

# where to save .pkl, after concatnate images
# PKL_PATH = r'E:\zcq\codes\pkl'
PKL_PATH = os.path.join(GRAND_PARENT_PATH, 'pkl')

PARENT_PATH = os.path.join(GRAND_PARENT_PATH, 'model/6-30')

# where to save dataset: self.raw_dataset, self.vocab, after load_data
RAW_DATASET_PATH = os.path.join(PARENT_PATH, 'raw_dataset.pkl')

VALID_DATASET_PATH =  os.path.join(PARENT_PATH, 'valid_dataset.pkl')

# saved image parent path, parent_path + gene_stage + .format, give the image file path
DATASET_PAR_PATH = PKL_PATH

# csvfile path
CSVFILE_PATH = os.path.join(GRAND_PARENT_PATH, 'csvfile.csv')

# original image parent path
IMAGE_PARENT_PATH = '/home/litiange/pic_data'

# tensorflow ckpt file, trained weight file
CKPT_PATH = os.path.join(GRAND_PARENT_PATH, 'vgg_16.ckpt')

# it is TOP 60
SAVE_PATH = os.path.join(PARENT_PATH, 'model.ckpt')
# MODEL_CKPT_PATH = os.path.join(PARENT_PATH, 'model.ckpt-5000')
MODEL_CKPT_PATH = None
# MODEL_CKPT_PATH = r'E:\zcq\codes\weakcnn\theano\6-10\model.ckpt-5000'

SAVE_RESULT_PATH = os.path.join(PARENT_PATH, 'result.pkl')