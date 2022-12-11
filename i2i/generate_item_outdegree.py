""" Create item out-degree for ML-1m dataset"""

from itertools import chain
from typing import Tuple

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from constants import (
    DATA_PATH,
    LINK_PREDICTION_MASK_RATE,
    LINK_PREDICTION_NEGATIVE_MULTIPLIER,
    LINK_PREDICTION_SEED,
    LINK_PREDICTION_VALIDATION_FILENAME,
    ML1M_SCHEMA,
    NEXT_ITEM_TIME_DIFF,
    OUTDEGREE_COUNTER_FILE_NAME,
    TRAIN_PARTITION,
    USE_LINK_PREDICTION,
)
from count_next_item import ItemOutDegreeCounter
from utils import get_spark, mask_relation_and_create_negative_pairs, read_ml1m


def load_data(spark: SparkSession) -> Tuple[SparkDataFrame, SparkDataFrame]:

    df_rating = read_ml1m(DATA_PATH)
    # create partition by year
    df_rating["partition"] = df_rating[ML1M_SCHEMA.ts].map(lambda x: x.year)
    sdf_rating_train = spark.createDataFrame(df_rating[df_rating.partition == TRAIN_PARTITION])
    sdf_rating_test = spark.createDataFrame(df_rating[df_rating.partition != TRAIN_PARTITION])
    return sdf_rating_train, sdf_rating_test


def get_counter(df: SparkDataFrame) -> pd.DataFrame:
    print("count item relation...")
    item_counter = ItemOutDegreeCounter()
    result = item_counter.count(df, ML1M_SCHEMA.uid, ML1M_SCHEMA.iid, ML1M_SCHEMA.ts, max_time_diff=NEXT_ITEM_TIME_DIFF)
    result = result.toPandas()
    result = result.set_index(ML1M_SCHEMA.iid)
    return result


def _mask_relation_and_write_validation_file(df: pd.DataFrame) -> pd.DataFrame:
    print("mask some edge to validation on link prediction task...")
    df, positive_but_masked_pairs, negative_pairs = mask_relation_and_create_negative_pairs(
        df, LINK_PREDICTION_MASK_RATE, LINK_PREDICTION_NEGATIVE_MULTIPLIER, LINK_PREDICTION_SEED
    )
    # write validation as file
    all_validation_pairs = chain(
        ((i1, i2, 1) for i1, i2 in positive_but_masked_pairs), ((i1, i2, 0) for i1, i2 in negative_pairs)
    )
    with open(LINK_PREDICTION_VALIDATION_FILENAME, "wt") as f:
        for i1, i2, label in all_validation_pairs:
            f.writelines("{}, {}, {}\n".format(i1, i2, label))
    return df


def main():
    spark = get_spark()
    train, _ = load_data(spark)
    train_df = get_counter(train)

    if USE_LINK_PREDICTION:
        train_df = _mask_relation_and_write_validation_file(train_df)

    train_df.to_csv(OUTDEGREE_COUNTER_FILE_NAME)


if __name__ == "__main__":
    main()
