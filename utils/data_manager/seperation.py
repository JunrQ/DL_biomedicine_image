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

    def __init__(self, config):
        self.stages = config.stages
        self.directions =config.directions
        self.annotation_threshold = config.annotation_number
        self.proportion = config.proportion
        self.tolerance_margin = config.tolerance_margin
        self.shuffle = config.shuffle_separation

    def separate(self, image_table, annot_table):
        """ Separate data set into train, validation and test set.

        Args:
            image_table: Pandas data frame of image set.
            annot_table: Pandas data frame of annotation set.
            proportion (dict): How to separate (e.g. {'train': 0.6, 'val': 0.1, 'test': 0.2}).
            tolerance_margin: Relax sample upper bound to percentage + tolerance_margin.
            shuffle: Add randomness.

        Return: A tuple. This first is an object with three attributes: train, validation, and test.
            The second field is the extracted vocabulary. And the third field is a list of ratios 
            of positive samples in each label.


        TODO: Instead of counting group number, try to count image number.

        """

        DataSep = namedtuple('DataSep', ['train', 'validation', 'test'])

        image_table, annot_table, vocab = self._shrink_data_set(
            image_table, annot_table)

        merged_table = self._merge_image_and_annot(image_table, annot_table)
        
        test_set, remain = self._seperate_one_part(
            merged_table, self.proportion['test'], self.tolerance_margin)

        if self.shuffle:
            remain = remain.sample(frac=1)
        # rebalance proportion
        train_proportion = self.proportion['train'] / (self.proportion['train'] + self.proportion['val'] )
        train_set, remain = self._seperate_one_part(
            remain, train_proportion, self.tolerance_margin)

        self._log_separation(train_set, remain, test_set)
        return DataSep(train=train_set, validation=remain, test=test_set), vocab
    
    def _log_separation(self, train_set, val_set, test_set):
        img_nums = seq((train_set, val_set, test_set)) \
            .map(self._unroll) \
            .map(len) \
            .list()
        
        print("Group numbers:")
        print(f"    train: {len(train_set)}, validation: {len(val_set)}, test: {len(test_set)}")
        print("Image numbers:")
        print(f"    train: {img_nums[0]}, validation: {img_nums[1]}, test: {img_nums[2]}")
        
    def _seperate_one_part(self, table, percentage, tolerance_margin):
        annot_stat_dict = self._build_annot_statistic(table)
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
        reversed_map = self._reverse_annot_map(table)
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

    def _merge_image_and_annot(self, image_table, annot_table):
        return pd.concat([image_table, annot_table], axis=1)
    
    def _unroll(self, series):
        return seq(series).flat_map(lambda l: l).list()
        
    def _reverse_annot_map(self, annot_table):
        unrolled = self._unroll(annot_table.annotation)
        unique = pd.Series(unrolled).unique()
        reverse_map = seq(unique).map(lambda annot: (annot, [])).dict()

        for group, row in annot_table.iterrows():
            for annot in row.annotation:
                reverse_map[annot].append(group)

        return reverse_map

    def _build_annot_statistic(self, annot_table):
        unrolled = self._unroll(annot_table.annotation)
        annot_sum = pd.Series(unrolled).value_counts()
        statistic = seq(annot_sum.iteritems()) \
            .smap(lambda annot, sum: (annot, (0, sum))) \
            .dict()
        return statistic

    def _extract_directions(self, image_table, annot_table, directions):
        image_remain = image_table[image_table.direction.isin(directions)]
        grouped = image_remain.groupby(['gene', 'stage'])
        merged = grouped.image_url.agg(
            lambda l: tuple([i for sl in l for i in sl]))
        drop_empty_mask = merged.apply(lambda urls: len(urls) > 0)
        merged = merged[drop_empty_mask]
        annot_remain = annot_table.loc[merged.index.values]

        return merged, annot_remain

    def _extract_stages(self, image_table, annot_table, stages):
        # multiindex column can not be selected
        # we need to convert a multiindex column to a normal column first
        image_table = image_table.reset_index()
        annot_table = annot_table.reset_index()
        image_remain = image_table[image_table.stage.isin(stages)]
        annot_remain = annot_table[annot_table.stage.isin(stages)]

        # and convert back to index
        return (image_remain.set_index(['gene', 'stage']),
                annot_remain.set_index(['gene', 'stage']))

    def _extract_vocabulary(self, annot_table, keep_number):
        unrolled = seq(annot_table.annotation) \
            .flat_map(lambda annots: annots) \
            .list()
        unrolled = pd.Series(unrolled)
        vocab = unrolled.value_counts().nlargest(keep_number).index.values

        return vocab

    def _drop_annot(self, annots, vocab):
        return tuple(seq(annots).filter(lambda a: a in set(vocab)).list())

    def _shrink_data_set(self, image_table, annot_table):
        image_table, annot_table = self._extract_stages(
            image_table, annot_table, self.stages)

        image_table, annot_table = self._extract_directions(
            image_table, annot_table, self.directions)

        vocab = self._extract_vocabulary(
            annot_table, self.annotation_threshold)
        annot_table.annotation = annot_table.annotation.apply(
            lambda annots: self._drop_annot(annots, vocab))

        drop_empty_mask = annot_table.annotation.apply(
            lambda annots: len(annots) > 0)
        dropped_annot = annot_table[drop_empty_mask]
        dropped_image = image_table.loc[dropped_annot.index.values]

        return dropped_image, dropped_annot, vocab
