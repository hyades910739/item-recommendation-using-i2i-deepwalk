import random
from collections import Counter
from datetime import datetime
from functools import reduce
from itertools import chain, groupby
from operator import itemgetter
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from tqdm import tqdm

from constants import EDGE_PAIR, ML1M_SCHEMA


def get_spark():
    spark = (
        SparkSession.builder.appName("deepwalk-example")
        .master("local[3]")
        .enableHiveSupport()
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )
    return spark


def read_ml1m(path: str, min_rate=3.1) -> pd.DataFrame:
    ratings = []
    with open(path, "rt") as f:
        for l in tqdm(f):
            uid, iid, rating, ts = l.strip().split("::")
            if int(rating) >= min_rate:
                ratings.append([uid, iid, int(ts)])

    df = pd.DataFrame(ratings, columns=[ML1M_SCHEMA.uid, ML1M_SCHEMA.iid, ML1M_SCHEMA.ts])
    df[ML1M_SCHEMA.ts] = df[ML1M_SCHEMA.ts].map(datetime.utcfromtimestamp)
    return df


def dic_normalize(dic):
    s = sum(dic.values())
    return {k: v / s for k, v in dic.items()}


def get_sample_from_dic(dic, n):
    return list(np.random.choice(list(dic.keys()), p=list(dic.values()), size=int(n)))


def get_total_in_degree_counter(series):
    return series.map(lambda x: Counter(x)).sum()


def popn(row, col_sample, col_n):
    try:
        return [row[col_sample].pop() for _ in range(int(row[col_n]))]
    except IndexError:
        return []


def mask_relation_and_create_negative_pairs(
    df: pd.DataFrame, mask_ratio: float = 0.2, negative_multiplier=1, seed=123
) -> Tuple[pd.DataFrame, List[EDGE_PAIR], List[EDGE_PAIR]]:
    """mask and create negative for link prediction task

    Args:
        df (pd.DataFrame): with index as items, and column `counter` as out degree dict.
        mask_ratio: ratio of relation to mask.
        negative_multiplier: how many negative pair to generate, if set to 2,
                             there will be 2 * (number of pair masked) negative pairs.
    """
    np.random.seed(seed)

    assert df.index.dtype == df["counter"].dtype, "dtype differ!"
    # get all items:
    out_degree_sets: pd.Series = df["counter"].map(set)
    all_items_in_counter: set = reduce(lambda x, y: x | y, out_degree_sets)
    all_items_in_index = set(df.index)
    all_items = list(all_items_in_index | all_items_in_counter)

    # get all_pairs and sample masked pairs.
    positive_pairs_set, positive_but_masked_pairs = _get_positive_pair_set_and_sample_masked_pairs(df, mask_ratio)

    num_masked = len(positive_but_masked_pairs)

    # sample:
    print("start to create negative pairs...")
    num_negatives = int(num_masked * negative_multiplier)
    negative_pairs = _create_negative_pairs(all_items, num_negatives, positive_pairs_set)

    # mask positives:
    _mask_out_degree_in_df(df, positive_but_masked_pairs)
    return df, positive_but_masked_pairs, negative_pairs


def _mask_out_degree_in_df(df: pd.DataFrame, mask_pairs: List[EDGE_PAIR]) -> None:
    "delete out degree from df, this is a inplace operation."
    for in_item, pairs in groupby(sorted(mask_pairs, key=itemgetter(0)), key=itemgetter(0)):
        counter = df.at[in_item, "counter"]
        for _, out_item in pairs:
            counter.pop(in_item, None)


def _create_negative_pairs(items: List[str], num_negatives: int, positive_pairs_set: Set[EDGE_PAIR]) -> List[EDGE_PAIR]:
    n_buffer = min(100_000, num_negatives * 2)
    negative_pairs = []
    while len(negative_pairs) < num_negatives:
        buffers = np.random.choice(items, size=(n_buffer, 2))
        for i1, i2 in buffers:
            if (i1, i2) not in positive_pairs_set:
                negative_pairs.append((i1, i2))
            if len(negative_pairs) >= num_negatives:
                break
    assert len(negative_pairs) == num_negatives
    return negative_pairs


def _get_positive_pair_set_and_sample_masked_pairs(
    df: pd.DataFrame, mask_ratio: float
) -> Tuple[Set[EDGE_PAIR], List[EDGE_PAIR]]:
    positive_pairs_set = set()
    positive_but_masked_pairs = []
    for item_in, item_outs in df["counter"].iteritems():
        for i in item_outs:
            positive_pairs_set.add((item_in, i))
            if random.random() <= mask_ratio:
                positive_but_masked_pairs.append((item_in, i))
    return positive_pairs_set, positive_but_masked_pairs
