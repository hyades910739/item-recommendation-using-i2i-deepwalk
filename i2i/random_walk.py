from typing import Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import get_sample_from_dic


class Sampler:
    "base class of sampler"

    def next(self):
        raise NotImplementedError()


def get_sample_from_dic(dic, n):
    return list(np.random.choice(list(dic.keys()), p=list(dic.values()), size=int(n)))


class MixSampler(Sampler):
    """
    Mix sample by popularity and get sample by index.
    """

    def __init__(self, all_items, prob_dict, uniform_ratio=0.05, buffer_size=100000):

        self.all_items = all_items
        self.prob_dict = prob_dict
        self.uniform_ratio = uniform_ratio
        self.buffer_size = int(buffer_size)

        self.buffer_samples = []
        self.uniform_iter = self.all_items.__iter__()

    def _get_prob_sample(self):
        try:
            x = self.buffer_samples.pop()
        except IndexError:
            self.buffer_samples = get_sample_from_dic(self.prob_dict, n=self.buffer_size)
            x = self.buffer_samples.pop()
        return x

    def _get_uniform_sample(self):
        try:
            x = next(self.uniform_iter)
        except StopIteration:
            self.uniform_iter = self.all_items.__iter__()
            x = next(self.uniform_iter)
        return x

    def next(self):
        if np.random.random() <= self.uniform_ratio:
            return self._get_uniform_sample()
        else:
            return self._get_prob_sample()


def random_walks(df: pd.DataFrame, start_sampler: Sampler, n_walk=1e6, stop_prob=0.05) -> List[List[Any]]:
    """
    Conduct random walk on counter df.

    df schema:
        samples: column that store samples as list
        n_sample: number of buffer sample in samples column
        counter: column that store dict of item:prob .
    """
    walks = []

    pbar = tqdm(total=n_walk, leave=True, position=0)
    cnt = 0
    while cnt < n_walk:

        this = str(start_sampler.next())
        walk = [this]
        while np.random.random() > stop_prob:
            try:
                next_ = df.at[this, "samples"].pop()
            except IndexError:
                df.at[this, "samples"] = get_sample_from_dic(df.at[this, "counter"], df.at[this, "n_sample"])
                next_ = df.at[this, "samples"].pop()
            walk.append(str(next_))
            this = next_
        walks.append(walk)
        pbar.update()
        cnt += 1

    return walks
