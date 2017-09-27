import pandas as pd
from functional import seq


class SeparationScheme(object):
    def __init__(self, annotation_threshold=40, image_threshold=10):
        self.annotation_threshold = annotation_threshold
        self.image_threshold = image_threshold

    def separate(self, image_table, annot_table):
        pass

    def extract_vocabulary(self, annot_table, keep_number):
        unrolled = seq(annot_table.annotation) \
            .flat_map(lambda annots: annots) \
            .list()
        unrolled = pd.Series(unrolled)
        vocab = set(unrolled.value_counts().nlargest(keep_number).index)
        return vocab

    def drop_annot(self, annots, vocab):
        return seq(annots).filter(lambda a: a in vocab).list()

    def shrink_data_set(self, image_table, annot_table):
        vocab = self.extract_vocabulary(annot_table, self.annotation_threshold)
        annot_table.annotation = annot_table.annotation.apply(
            lambda annots: self.drop_annot(annots, vocab))
        
        drop_empty_mask = annot_table.annotation.apply(lambda annots: len(annots) > 0)
        dropped_annot = annot_table[drop_empty_mask]
        dropped_image = image_table.loc[annot_table.index.values, :]

        return dropped_image, dropped_annot
