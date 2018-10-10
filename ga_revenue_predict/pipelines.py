"""
Pipelines and transforms to be performed.
"""
import logging
from sklearn.base import TransformerMixin

# Custom transformers for this project only


class RemoveColumns(TransformerMixin):
    """
    Drop unused columns from train and test set.
    """
    def __init__(self, use_col):
        self.use_col = use_col

    def transform(self, X):
        # Check if any use_col does not exist in DF, and omit them
        col_omit = []
        for col in self.use_col:
            if col not in X.columns:
                logging.warning("Column {} not found in DataFrame. Omit.".format(col))
                col_omit.append(col)
        for col in col_omit:
            self.use_col.remove(col)
        return X[self.use_col]


class ColumnSelector(TransformerMixin):
    def __init__(self, col):
        self.col = col

    def transform(self, X):
        return X[self.col]


class generate_datetime_features(TransformerMixin):
    def transform(self, X):
        """
        Generates following features from `date` object `X`:
        YEAR, MONTH, DAY OF YEAR, DAY OF MONTH, WEEK OF YEAR, WEEKDAY
        if `X` is `datetime` object, the following extra features are returned:
        HOUR, MINUTE, MINUTE OF DAY
        :param X:
        :return:
        """
        pass
