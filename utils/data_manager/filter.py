# -*- coding: utf-8 -*-
""" Filter image table and annotation table by config object.
"""

from functional import seq


def filter_stages_and_directions(image_table, annot_table, stages, directions):
    """ Filter data set by its stages and directions.

    Args:
        image_table (DataFrame): A `DataFrame` whose index is gene and stage (
            multiindex), and has a column named as `image_url`.
        annot_table (DataFrame): A `DataFrame` whose index is gene and stage,
            and has a column named as `annotation`.
        stages (iterable): What stages to keep.
        directions (iteratble): What directions to keep.

    Return:
        Shrinked image table and annotation table.

    Note:
        If `image_table` does not contain a column named as `direction`, the
    `direction` field in `config` will be omitted.
    """
    imgs, annots = _filter_stages(image_table, annot_table, stages)
    imgs, annots = _filter_directions(imgs, annots, directions)

    return imgs, annots


def filter_labels(image_table, annot_table, vocab):
    """ Filter data set by vocabulary. Only the labels in the vocabulary will
    be kept.

    Args:
        image_table (DataFrame): A `DataFrame` whose index is gene and stage (
            multiindex), and has a column named as `image_url`.
        annot_table (DataFrame): A `DataFrame` whose index is gene and stage,
            and has a column named as `annotation`.
        vocab (iterable): Vocabulary of labels.

    Return:
        Shrinked image table and annotation table.

    """
    def drop_annot(annots, vocab):
        """ filter a group of labels by vocab.
        """
        vocab = set(vocab)
        return tuple(seq(annots).filter(lambda a: a in vocab).list())

    dropped_annot = annot_table.copy(deep=True)
    dropped_annot.annotation = annot_table.annotation.apply(
        lambda annots: drop_annot(annots, vocab))
    nonempty_mask = dropped_annot.annotation.apply(
        lambda annots: len(annots) > 0)
    dropped_annot = dropped_annot[nonempty_mask]
    dropped_image = image_table.loc[dropped_annot.index.values]

    return dropped_image, dropped_annot


def _filter_stages(image_table, annot_table, stages):
    # multiindex column can not be selected
    # we need to convert a multiindex column to a normal column first
    image_table = image_table.reset_index()
    annot_table = annot_table.reset_index()
    image_remain = image_table[image_table.stage.isin(stages)]
    annot_remain = annot_table[annot_table.stage.isin(stages)]

    # and convert back to index
    return (image_remain.set_index(['gene', 'stage']),
            annot_remain.set_index(['gene', 'stage']))


def _filter_directions(image_table, annot_table, directions):
    if 'direction' not in image_table.columns:
        return image_table, annot_table

    image_remain = image_table[image_table.direction.isin(directions)]
    grouped = image_remain.groupby(['gene', 'stage'])
    merged = grouped.image_url.aggregate(
        lambda l: tuple([i for sl in l for i in sl]))
    nonempty_mask = merged.apply(lambda urls: len(urls) > 0)
    merged = merged[nonempty_mask]
    annot_remain = annot_table.loc[merged.index.values]

    return merged, annot_remain
