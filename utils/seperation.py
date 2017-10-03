# -*- coding: utf-8 -*-
""" This module defines a class that separate data set fairly 
for training and testing.
"""

from collections import namedtuple
from functional import seq
import pandas as pd

class SeparationScheme(object):
    """ Separate data set into train, validation, and test set.
    """

    def __init__(self, stages=None, directions=None, annotation_threshold=40, image_threshold=10):
        if stages is None:
            stages = [1, 2, 3, 4, 5, 6]
        if directions is None:
            directions = ['dorsal', 'lateral', 'ventral']
        self.stages = stages
        self.directions = directions
        self.annotation_threshold = annotation_threshold
        self.image_threshold = image_threshold

    def separate(self, image_table, annot_table, proportion, tolerance_margin=0.02):
        """ Separate data set into train, validation and test set.

        Args:
            image_table: Pandas data frame of image set.
            annot_table: Pandas data frame of annotation set.
            proportion (dict): e.g. {'train': 0.6, 'val': 0.1, 'test': 0.2}

        Return: A namedtuple with three fields: train, validation, and test.

        """
        DataSep = namedtuple('DataSep', ['train', 'validation', 'test'])

        image_table, annot_table = self.__shrink_data_set(
            image_table, annot_table)

        merged_table = self.__merge_image_and_annot(image_table, annot_table)
        train_set, remain = self.__seperate_one_part(
            merged_table, proportion['train'], tolerance_margin)
        val_set, remain = self.__seperate_one_part(
            remain, proportion['val'], tolerance_margin)

        return DataSep(train=train_set, validation=val_set, test=remain)

    def __seperate_one_part(self, table, percentage, tolerance_margin):
        annot_stat_dict = self.__build_annot_statistic(table)
        selected_group_indices = []
        selected_set = set()

        for group, row in table.iterrows():
            choosed = True
            for annot in row.annotation:
                covered_num, total_num = annot_stat_dict[annot]
                # exceed limit
                if (covered_num + 1) / total_num > percentage:
                    choosed = False
                    break

            if choosed:
                selected_group_indices.append(group)
                selected_set.add(group)
                # update annotation statistics
                for annot in row.annotation:
                    covered_num, total_num = annot_stat_dict[annot]
                    annot_stat_dict[annot] = (covered_num + 1, total_num)

        # relax the percentage restriction to percentage + tolerance_margin
        reversed_map = self.__reverse_annot_map(table)
        # repeat (fixed point algorithm)
        for _ in range(0, len(reversed_map)):
            for annot in reversed_map.keys():
                covered_num, total_num = annot_stat_dict[annot]
                # already collected enough samples
                if (covered_num / total_num) >= percentage:
                    continue

                # else need more samples
                # iterate through associated groups
                for group in reversed_map[annot]:
                    # discard already selected groups
                    if group in selected_set:
                        continue
                    # check if it is proper to choose this group
                    proper = True
                    for associated_annot in table.loc[group].annotation:
                        covered_num, total_num = annot_stat_dict[associated_annot]
                        # using relaxed upper bound
                        if (covered_num + 1) / total_num > percentage + tolerance_margin:
                            proper = False
                            break
                    # if proper
                    if proper:
                        # collect indices and update selected_set
                        selected_group_indices.append(group)
                        selected_set.add(group)
                        # update annotation statistics
                        for associated_annot in table.loc[group].annotation:
                            covered_num, total_num = annot_stat_dict[associated_annot]
                            annot_stat_dict[associated_annot] = (
                                covered_num + 1, total_num)

        return table.loc[selected_group_indices, :], \
            table.loc[~table.index.isin(selected_group_indices)]

    def __merge_image_and_annot(self, image_table, annot_table):
        return pd.concat([image_table, annot_table], axis=1)

    def __reverse_annot_map(self, annot_table):
        unrolled = seq(annot_table.annotation) \
            .flat_map(lambda annots: annots) \
            .list()
        unique = pd.Series(unrolled).unique()
        reverse_map = seq(unique).map(lambda annot: (annot, [])).dict()

        for group, row in annot_table.iterrows():
            for annot in row.annotation:
                reverse_map[annot].append(group)

        return reverse_map

    def __build_annot_statistic(self, annot_table):
        unrolled = seq(annot_table.annotation) \
            .flat_map(lambda annots: annots) \
            .list()
        annot_sum = pd.Series(unrolled).value_counts()
        statistic = seq(annot_sum.iteritems()) \
            .smap(lambda annot, sum: (annot, (0, sum))) \
            .dict()
        return statistic

    def __extract_directions(self, image_table, annot_table, directions):
        image_remain = image_table[image_table.direction.isin(directions)]
        grouped = image_remain.groupby(['gene', 'stage'])
        merged = grouped.image_url.agg(
            lambda l: tuple([i for sl in l for i in sl]))
        drop_empty_mask = merged.apply(lambda urls: len(urls) > 0)
        merged = merged[drop_empty_mask]
        annot_remain = annot_table.loc[merged.index.values]

        return merged, annot_remain

    def __extract_stages(self, image_table, annot_table, stages):
        # multiindex column can not be selected
        # we need to convert a multiindex column to a normal column first
        image_table = image_table.reset_index()
        annot_table = annot_table.reset_index()
        image_remain = image_table[image_table.stage.isin(stages)]
        annot_remain = annot_table[annot_table.stage.isin(stages)]

        # and convert back to index
        return (image_remain.set_index(['gene', 'stage']),
                annot_remain.set_index(['gene', 'stage']))

    def __extract_vocabulary(self, annot_table, keep_number):
        unrolled = seq(annot_table.annotation) \
            .flat_map(lambda annots: annots) \
            .list()
        unrolled = pd.Series(unrolled)
        vocab = set(unrolled.value_counts().nlargest(keep_number).index)
        return vocab

    def __drop_annot(self, annots, vocab):
        return tuple(seq(annots).filter(lambda a: a in vocab).list())

    def __shrink_data_set(self, image_table, annot_table):
        image_table, annot_table = self.__extract_stages(
            image_table, annot_table, self.stages)

        image_table, annot_table = self.__extract_directions(
            image_table, annot_table, self.directions)

        vocab = self.__extract_vocabulary(
            annot_table, self.annotation_threshold)
        annot_table.annotation = annot_table.annotation.apply(
            lambda annots: self.__drop_annot(annots, vocab))

        drop_empty_mask = annot_table.annotation.apply(
            lambda annots: len(annots) > 0)
        dropped_annot = annot_table[drop_empty_mask]
        dropped_image = image_table.loc[dropped_annot.index.values]


        return dropped_image, dropped_annot