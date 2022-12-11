import pickle
from collections import Counter
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from constants import ITEM_MIN_INDEGREE, ML1M_SCHEMA, N_RANDOM_WALK, OUTDEGREE_COUNTER_FILE_NAME, RANDOM_WALK_FILE_NAME
from random_walk import MixSampler, random_walks
from utils import dic_normalize


def _load_in_degree_df() -> pd.DataFrame:
    df = pd.read_csv(
        OUTDEGREE_COUNTER_FILE_NAME,
    )
    df = df.set_index(ML1M_SCHEMA.iid)
    df["counter"] = df.counter.map(eval)  # back to dict format
    df.index = df.index.map(str)
    return df


def _filter_items(df: pd.DataFrame) -> Tuple[pd.DataFrame, Counter]:
    indegree_counter = _get_total_indegrees(df["counter"])
    items_been_filter_out = {k for k, v in indegree_counter.items() if v <= ITEM_MIN_INDEGREE}

    indegree_counter = {k: v for k, v in indegree_counter.items() if v > ITEM_MIN_INDEGREE}
    new_idx = set(df.index) - items_been_filter_out
    new_idx = new_idx & indegree_counter.keys()  # filter out no in_degree items
    df = df.loc[new_idx, :]
    # remove item in dict for every row
    df["counter"] = df.counter.map(lambda d: {k: v for k, v in d.items() if k not in items_been_filter_out})

    return df, indegree_counter


def _get_total_indegrees(counters: pd.Series) -> Counter:
    return counters.map(lambda x: Counter(x)).sum()


def main():

    df = _load_in_degree_df()
    df, indegree_counter = _filter_items(df)
    indegree_counter_normalize = dic_normalize(indegree_counter)
    # normalize counter
    df["counter"] = df["counter"].map(dic_normalize)
    # create n_sample column: indegree_counter multipy by 5
    df["n_sample"] = df.index.map(indegree_counter) * 5
    # init samples column
    df["samples"] = [[] for _ in range(df.shape[0])]

    # conduct random walk:
    sampler = MixSampler(list(df.index), indegree_counter_normalize, buffer_size=100_000)
    walks = random_walks(df, sampler, n_walk=N_RANDOM_WALK)
    # save txt:
    with open(RANDOM_WALK_FILE_NAME, "wt") as f:
        for walk in tqdm(walks):
            f.writelines(" ".join(walk))
            f.writelines("\n")


if __name__ == "__main__":
    main()
