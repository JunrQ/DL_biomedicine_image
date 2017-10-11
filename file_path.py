import os

# where to save .pkl, after concatnate images
PKL_PATH = r'E:\zcq\codes\pkl'

PARENT_PATH = r'E:\zcq\codes\weakcnn\theano\6-10'

# where to save dataset: self.raw_dataset, self.vocab, after load_data
RAW_DATASET_PATH = os.path.join(PARENT_PATH, 'raw_dataset.pkl')

VALID_DATASET_PATH =  os.path.join(PARENT_PATH, 'valid_dataset.pkl')

# saved image parent path, parent_path + gene_stage + .format, give the image file path
DATASET_PAR_PATH = PKL_PATH

# csvfile path
CSVFILE_PATH = r'E:\csvfile.csv'

# original image parent path
IMAGE_PARENT_PATH = r'E:\pic_data'

# tensorflow ckpt file, trained weight file
CKPT_PATH = r'E:\zcq\codes\weakcnn\theano\vgg_16.ckpt'

# it is TOP 60
SAVE_PATH = os.path.join(PARENT_PATH, 'model.ckpt')
# MODEL_CKPT_PATH = os.path.join(PARENT_PATH, 'model.ckpt-5000')
MODEL_CKPT_PATH = None
# MODEL_CKPT_PATH = r'E:\zcq\codes\weakcnn\theano\6-10\model.ckpt-5000'

SAVE_RESULT_PATH = os.path.join(PARENT_PATH, 'result.pkl')