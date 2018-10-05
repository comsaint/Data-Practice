"""
Class for data reading.
"""

import sys
import json
import pandas as pd
from pandas.io.json import json_normalize
import logging
from .Settings import USE_COLS


class Loader(object):
    def __init__(self):
        pass

    @staticmethod
    def read_original_data(table_code, preprocess=False, json_columns=None, **kwargs):
        if table_code is None:
            logging.error("No table code supplied.")
            sys.exit(-1)

        from .Paths import DICT_PATHS
        if table_code in DICT_PATHS.keys():
            # Read training data
            from .Schemas import SCHEMAS
            if table_code in SCHEMAS.keys():
                df = pd.read_csv(DICT_PATHS[table_code], dtype=SCHEMAS[table_code], **kwargs)
            else:
                logging.warning("No schema found for table '{}', default dtypes to STR.".format(table_code))
                df = pd.read_csv(DICT_PATHS[table_code], dtype=str, **kwargs)

            if json_columns:
                for column in json_columns:
                    column_as_df = json_normalize(df[column])
                    column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
                    df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
            logging.info(f"Loaded {DICT_PATHS[table_code]}. Shape: {df.shape}")

        # Preprocess the data if necessary, e.g. date conversion, fillna etc.
        if preprocess is True:
            df = df[USE_COLS]

        return df

    def run(self, mode, subsample=None):
        logging.info("Running Loader...")

        if mode == 'raw':
            JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
            karg = {
                'converters': {column: json.loads for column in JSON_COLUMNS},
                'parse_dates': ['date'],
                'infer_datetime_format': True
            }
            if subsample is not None:
                import random
                karg.update({'header': 0,
                             'skiprows': lambda i: i > 0 and random.random() > subsample})

            logging.info("Loading raw, train data...")
            df_train = self.read_original_data('train', preprocess=False, json_columns=JSON_COLUMNS, **karg)
            logging.info("Loading raw, test data...")
            df_test = self.read_original_data('test', preprocess=False, json_columns=JSON_COLUMNS, **karg)
        elif mode == 'parsed':
            karg = {
                'parse_dates': ['date'],
                'infer_datetime_format': True,
                'usecols': USE_COLS
            }
            if subsample is not None:
                import random
                karg.update({'header': 0,
                             'skiprows': lambda i: i > 0 and random.random() > subsample})
            logging.info("Loading parsed, train data...")
            df_train = self.read_original_data('train_parsed', preprocess=False, **karg)
            logging.info("Loading parsed, test data...")
            karg['usecols'].remove('totals.transactionRevenue')
            df_test = self.read_original_data('test_parsed', preprocess=False, **karg)
        else:
            logging.error("Unrecognized mode: '{}'".format(mode))
            sys.exit(-1)
        return df_train, df_test


if __name__ == '__main__':
    loader = Loader()
    loader.run(mode='raw')
