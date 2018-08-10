"""
Class for data reading.
"""

import sys
import pandas as pd
import logging


class Loader(object):
    def __init__(self):
        pass

    @staticmethod
    def read_original_data(table_code, preprocess=True):
        if table_code is None:
            logging.error("No table code supplied.")
            sys.exit(-1)

        from .Paths import DICT_PATHS
        if table_code in DICT_PATHS.keys():
            # Read training data
            from .Schemas import SCHEMAS
            if table_code in SCHEMAS.keys():
                df = pd.read_csv(DICT_PATHS[table_code], dtype=SCHEMAS[table_code])
            else:
                logging.warning("No schema found for table '{}', default dtypes to STR.".format(table_code))
                df = pd.read_csv(DICT_PATHS[table_code], dtype=str)

        # Preprocess the data if necessary, e.g. date conversion, fillna etc.
        if preprocess is True:
            pass

        return df

    def run(self):
        logging.info("Running Loader...")
        pass


if __name__ == '__main__':
    loader = Loader()
    loader.run()
