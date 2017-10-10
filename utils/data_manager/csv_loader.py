# -*- coding: utf-8 -*-
""" Load image and annotation from csv file.

The csv files contain nontrivial index columns and special cells
that contain tuple. So we need a dedicated loader.
"""

import ast
import pandas as pd

def __generic(field):
    return ast.literal_eval(field)


def load_image_table(file_path):
    """ Load image table from csv file.
    """
    return pd.read_csv(file_path, index_col=[0, 1],
                       converters={'image_url': __generic})


def load_annot_table(file_path):
    """ Load annotation table from csv file.
    """
    return pd.read_csv(file_path, index_col=[0, 1],
                       converters={'annotation': __generic})
