# -*- coding: utf-8 -*-
""" This module defines a class that separate data set fairly
for training and testing.
"""

from collections import namedtuple
from functional import seq
import pandas as pd

class DataSep(namedtuple('DataSep', ['train', 'validation', 'test'])):
    """ Nametuple contains separated data set.
    """
    pass


def separate(image_table, annot_table, config, random_seed):
    """ Separate data set into train, validation and test set.

    Args:
        image_table: Pandas data frame of image set.
        annot_table: Pandas data frame of annotation set.
        proportion (dict): How to separate (e.g. {'train': 0.6, 'val': 0.1, 'test': 0.2}).
        tolerance_margin: Relax sample upper bound to percentage + tolerance_margin.
        shuffle: Add randomness.

    Return: {'train': train_set, 'val': validation_set, 'test': test_set}
    """
    merged_table = _merge_image_and_annot(image_table, annot_table)

    if config.shuffle_separation:
        merged_table = merged_table.sample(frac=1, random_state=random_seed)

    test_set, remain = _seperate_one_part(
        merged_table, config.proportion['test'], config.tolerance_margin)
    # rebalance proportion
    # avoid divide-by-zero error
    if config.proportion['train'] + config.proportion['val'] == 0:
        train_proportion = 0.0
    else:
        train_proportion = config.proportion['train'] / \
            (config.proportion['train'] + config.proportion['val'])

    train_set, remain = _seperate_one_part(
        remain, train_proportion, config.tolerance_margin)

    #_log_separation(train_set, remain, test_set)
    return DataSep(train=train_set, validation=remain, test=test_set)


def _log_separation(train_set, val_set, test_set):
    img_nums = seq((train_set, val_set, test_set)) \
        .map(lambda s: _unroll(s.image_url)) \
        .map(len) \
        .list()

    print(
        f'''Group numbers:
train: {len(train_set)}, validation: {len(val_set)}, test: {len(test_set)}
Image numbers:
train: {img_nums[0]}, validation: {img_nums[1]}, test: {img_nums[2]}
''')


def _seperate_one_part(table, percentage, tolerance_margin):
    annot_stat_dict = _build_annot_statistic(table)
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
    reversed_map = _reverse_annot_map(table)
    # repeat (fixed point algorithm)
    for _ in range(0, 1):
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


def _merge_image_and_annot(image_table, annot_table):
    return pd.concat([image_table, annot_table], axis=1)


def _unroll(series):
    return seq(series).flat_map(lambda l: l).list()


def _reverse_annot_map(annot_table):
    unrolled = _unroll(annot_table.annotation)
    unique = pd.Series(unrolled).unique()
    reverse_map = seq(unique).map(lambda annot: (annot, [])).dict()

    for group, row in annot_table.iterrows():
        for annot in row.annotation:
            reverse_map[annot].append(group)

    return reverse_map


def _build_annot_statistic(annot_table):
    unrolled = _unroll(annot_table.annotation)
    annot_sum = pd.Series(unrolled).value_counts()
    statistic = seq(annot_sum.iteritems()) \
        .smap(lambda annot, sum: (annot, (0, sum))) \
        .dict()
    return statistic
