import pyspark.sql.functions as F
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.window import Window


class ItemOutDegreeCounter:
    "count number of next item relation as counter column and return a spark dataframe"

    def __init__(self):
        self.prev_iid = "prev_iid"
        self.prev_ts = "prev_ts"
        self.is_valid_timediff = "is_valid_timediff"
        self.key = "key"
        self.val = "val"
        self.counter = "counter"

    def count(self, df: SparkDataFrame, uid: str, iid: str, ts: str, max_time_diff: int) -> SparkDataFrame:
        """
        df: a df with column:
            uid: user id column
            iid: item id column
            ts: timestamp column
        max_time_diff: if time difference between two items exceed this value, the degree wont be counted.
        """
        self._check_schema(df, uid, iid, ts)
        df = self._check_or_parse_ts_column(df, ts)
        df = self._create_lag1_column(df, uid, iid, ts)
        df = self._create_valid_time_diff_col(df, ts, self.prev_ts, max_time_diff)
        df = self._filter_invalid_row(df)
        df = self._groupby_and_count_next_iid_as_counter(df, iid)
        df = self._filter_and_rename_column(df, iid)

        return df

    def _check_schema(self, df: SparkDataFrame, uid: str, iid: str, ts: str) -> None:
        # check column exist
        for col in [uid, iid, ts]:
            assert col in df.columns

        # check column name not in df:
        preserved_col_names = [self.prev_iid, self.prev_ts, self.is_valid_timediff, self.key, self.val, self.counter]
        for col_name in preserved_col_names:
            if col_name in df.columns:
                # change column name
                setattr(self, col_name, col_name + "_special_suffix")

    def _check_or_parse_ts_column(self, df: SparkDataFrame, ts: str) -> SparkDataFrame:
        _, dtype = df.select(ts).dtypes[0]
        if dtype == "timestamp":
            # to unix timestamp:
            df = df.withColumn(ts, F.unix_timestamp(ts))
        elif dtype in ("double", "bigint"):
            pass
        else:
            raise ValueError(f"Invalid timestamp type: {ts}")
        return df

    def _create_lag1_column(self, df, uid, iid, ts) -> SparkDataFrame:
        "use window function to get lag1 item and timestamp"
        windowspec = Window.partitionBy(uid).orderBy(ts)
        df = df.withColumn(self.prev_iid, F.lag(iid).over(windowspec)).withColumn(
            self.prev_ts, F.lag(ts).over(windowspec)
        )
        return df

    def _create_valid_time_diff_col(self, df: SparkDataFrame, col1: str, col2: str, maxtimediff: int) -> SparkDataFrame:
        # time_diff = F.unix_timestamp(col1) - F.unix_timestamp(col2)
        time_diff = F.col(col1) - F.col(col2)
        valid_td = time_diff < maxtimediff
        df = df.withColumn(self.is_valid_timediff, valid_td)
        return df

    def _filter_invalid_row(self, df: SparkDataFrame) -> SparkDataFrame:
        "filter first item in the session and interaction which over max time difference"
        return df.filter(~F.isnull(self.prev_iid) & F.col(self.is_valid_timediff))

    def _groupby_and_count_next_iid_as_counter(self, df: SparkDataFrame, iid: str) -> SparkDataFrame:
        df = df.groupby(self.prev_iid, iid).count()
        # create counter
        df = df.groupby(self.prev_iid).agg(F.collect_list(iid).alias(self.key), F.collect_list("count").alias(self.val))
        df = df.withColumn(self.counter, F.map_from_arrays(self.key, self.val))
        return df

    def _filter_and_rename_column(self, df: SparkDataFrame, iid: str) -> SparkDataFrame:
        df = df.select(F.col(self.prev_iid).alias(iid), self.counter)
        return df
