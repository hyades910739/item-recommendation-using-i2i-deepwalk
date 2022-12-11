from typing import Dict

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score

l2 = lambda x: np.sqrt((x**2).sum())


def link_prediction_evaluate(df: pd.DataFrame, model: Word2Vec, oov: str = "skip") -> Dict[str, float]:
    """Calculate auc for link prediction task.

    Args:
        oov: whether skip or  use_avg
        model: word2vec model instance.
        df: data with following column:
          * in_item: in item id
          * out_item: out item id
          * label: [0,1]. linked or not.
    Returns:
        metric dict
    """
    """
       Calculate auc for link prediction task.
       args:
       oov: whether skip or  use_avg
       df: data with following column:
         * in_item: in item id (str type)
         * out_item: out item id (str type)
         * label: [0,1]. linked or not.

    """
    assert oov in ("skip", "use_avg")
    for col in ["in_item", "out_item", "label"]:
        assert col in df.columns

    #
    df["in_vecs"] = df["in_item"].map(lambda w: _get_vector(model, w))
    df["out_vecs"] = df["out_item"].map(lambda w: _get_vector(model, w))
    nulls = df["in_vecs"].isnull() | df["out_vecs"].isnull()

    dot_products = (df["in_vecs"][~nulls] * df["out_vecs"][~nulls]).apply(sum)
    cosines = dot_products / (df["in_vecs"][~nulls].map(l2) * df["out_vecs"][~nulls].map(l2))
    df["cosines"] = cosines

    preds = df[~nulls]["cosines"]
    labels = df[~nulls]["label"]

    return {"auc": roc_auc_score(labels, preds), "null_rate": sum(nulls) / df.shape[0]}


def _get_vector(model, word, oov="pass"):
    try:
        vec = model.wv[word]
    except KeyError:
        if oov == "raise":
            raise
        else:
            return None
    return vec
