"""
Pipelines and transforms to be performed.
"""
import logging
from sklearn.base import TransformerMixin
from .Settings import USE_COLS

# Custom transformers for this project only


class RemoveColumns(TransformerMixin):
    """
    Drop unused columns from train and test set.
    """
    def transform(self, X):
        all_columns = X.columns
        if 'totals.transactionRevenue' not in all_columns:
            USE_COLS.remove('totals.transactionRevenue')
        col_omit = []

        # Check if any USE_COLS does not exist in DF, and omit them
        for col in USE_COLS:
            if col not in all_columns:
                logging.warning("Column {} not found in DataFrame. Omit.".format(col))
                col_omit.append(col)
        for col in col_omit:
            USE_COLS.remove(col)
        return X[USE_COLS]


class ColumnSelector(TransformerMixin):
    def __init__(self, col):
        self.col = col

    def transform(self, X):
        return X[self.col]


