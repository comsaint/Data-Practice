"""
Pipelines and transforms to be performed.
"""
import logging
from sklearn.base import TransformerMixin
import datetime
import pandas as pd
import numpy as np

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


class DatetimeFeaturesGenerator(TransformerMixin):
    def transform(self, X, y=None):
        """
        Generates following features from `date` object `X`:
        YEAR, MONTH, DAY, QUARTER, DAY OF YEAR, WEEK OF YEAR, WEEKDAY
        if `X` is `datetime` object, the following extra features are returned:
        HOUR, MINUTE, MINUTE OF DAY
        :param X:
        :return:
        """
        X_out = pd.concat([X.dt.year, X.dt.month, X.dt.day,
                           X.dt.quarter, X.dt.dayofyear, X.dt.weekofyear, X.dt.weekday], axis=1)
        X_out.columns = [X.name + '_' + n for n in [
            'year', 'month', 'day', 'quarter', 'dayofyear', 'weekofyear', 'weekday'
        ]]
        if X.dtype == np.dtype('datetime64[ns]'):
            X_out_time = pd.concat([X.dt.hour, X.dt.minute], axis=1)
            X_out_time.columns = [X.name + '_' + n for n in ['hour', 'minute']]
            X_out = pd.concat([X_out, X_out_time], axis=1)
        return X_out


class TimestampToDatetimeConverter(TransformerMixin):
    def __init__(self, tz=datetime.timezone.utc):
        self.tz = tz  # default UTC time

    def transform(self, X, y=None):
        return X.map(lambda t: datetime.datetime.fromtimestamp(t, tz=self.tz))


class PeriodicFeatureEncoder(TransformerMixin):
    def __init__(self, bound=None):
        self.bound = bound

    def fit(self, X, y=None):
        if self.bound is None:
            self.bound = [X.min(), X.max()]
        return self

    def transform(self, X, y=None):
        X_out = pd.concat([np.sin(2 * np.pi * X / float(self.bound[1] - self.bound[0])),
                           np.cos(2 * np.pi * X / float(self.bound[1] - self.bound[0]))], axis=1)
        X_out.columns = [X.name + '_' + n for n in ['sin', 'cos']]
        return X_out
