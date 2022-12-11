from typing import Tuple

from pydantic import BaseModel

EDGE_PAIR = Tuple[str, str]  # a edge represent (in_node, out_node), node should be a string.


# data process config
DATA_PATH = "../ml-1m/ratings.dat"
OUTDEGREE_COUNTER_FILE_NAME = "../files/ml1m_in_degree_counter_2week_in_2020.csv"
RANDOM_WALK_FILE_NAME = "../files/random_walks.txt"
NEXT_ITEM_TIME_DIFF = 14 * 24 * 60 * 60  # two weeks
TRAIN_PARTITION = 2000
ITEM_MIN_INDEGREE = 3


class ML1M_SCHEMA:
    uid = "uid"
    iid = "iid"
    ts = "ts"


# random walk config
N_RANDOM_WALK = 1_000_000

# Link prediction config
USE_LINK_PREDICTION = True
LINK_PREDICTION_SEED = 5566
LINK_PREDICTION_MASK_RATE = 0.2
LINK_PREDICTION_NEGATIVE_MULTIPLIER = 1
LINK_PREDICTION_VALIDATION_FILENAME = "../files/ml1m_link_prediction_validation.csv"

# i2i config
class W2VConfig(BaseModel):
    size: int = 128
    window: int = 7
    min_count: int = 1
    iter: int = 5
    negative: int = 5
    ns_exponent: float = 0
    seed: int = 5566
    workers: int = 6


class FaissConfig(BaseModel):
    topn: int = 100
    nlist: int = 100
    nprobe: int = 20


EVALUATE_AFTER_W2V = True

W2V_MODEL_NAME = "../files/w2v.model"
I2I_MAP_NAME = "../files/i2i_map.json"
