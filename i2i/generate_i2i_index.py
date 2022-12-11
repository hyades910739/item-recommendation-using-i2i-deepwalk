import json
from itertools import takewhile
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from constants import (
    EVALUATE_AFTER_W2V,
    I2I_MAP_NAME,
    LINK_PREDICTION_VALIDATION_FILENAME,
    RANDOM_WALK_FILE_NAME,
    W2V_MODEL_NAME,
    FaissConfig,
    W2VConfig,
)
from link_prediction import link_prediction_evaluate


def train_w2v_model(config: W2VConfig) -> Word2Vec:
    print("Train W2V: loading sentence...")
    ls = LineSentence(source=RANDOM_WALK_FILE_NAME)
    print("Train W2V: start to train word2vec embedding")
    model = Word2Vec(sentences=ls, sg=1, **config.dict())
    print("Train W2V: training complete.")
    return model


def create_i2i_index(model: Word2Vec, topn=100, nlist=100, nprobe=20) -> Dict[str, List[str]]:

    vectors = model.wv.vectors
    l2_norms = np.sqrt((vectors**2).sum(1))
    normalized_vectors = vectors / np.expand_dims(l2_norms, -1)
    _, d = normalized_vectors.shape

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    index.train(normalized_vectors)
    index.add(normalized_vectors)
    index.nprobe = nprobe
    _, items = index.search(normalized_vectors, k=topn)

    maps = dict()
    i2w = model.wv.index2word
    for this_item_idx, (this_item, item_list) in enumerate(zip(i2w, items)):
        # if i = -1, means that this item (and items afterward) is none.
        # i == this_item_idx means item itself (unnecessary in i2i situation)
        maps[this_item] = [i2w[i] for i in takewhile(lambda x: x != -1, item_list) if i != this_item_idx]

    return maps


def _validation(model: Word2Vec) -> Dict[str, float]:
    df = pd.read_csv(LINK_PREDICTION_VALIDATION_FILENAME, header=None)
    df.columns = ["in_item", "out_item", "label"]
    df["in_item"] = df["in_item"].map(str)
    df["out_item"] = df["out_item"].map(str)
    res = link_prediction_evaluate(df, model)
    return res


def main():
    w2v_config = W2VConfig()
    faiss_config = FaissConfig()
    model = train_w2v_model(w2v_config)
    model.save(W2V_MODEL_NAME)

    if EVALUATE_AFTER_W2V:
        print("EVALUATE: ...")
        metrics = _validation(model)
        print(f"result: {metrics}")

    print("faiss: create nearest item list.")
    maps = create_i2i_index(model, faiss_config.topn, faiss_config.nlist, faiss_config.nprobe)
    with open(I2I_MAP_NAME, "wt") as f:
        json.dump(maps, f)
    print("faiss: complete.")


if __name__ == "__main__":
    main()
