"""
This module stores often-used Sklearn utilities, for example column selector, categorical encoder, time extractor,
 etc.
Reusing this module is encouraged.
"""
import datetime
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin, BaseEstimator
#import matplotlib.pyplot as plt
#import itertools


class ColumnSelector(TransformerMixin):
    """
    Select `columns` from `X`.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]


class CategoricalEncoder(TransformerMixin):
    """
    Convert to Categorical with specific `categories`.

    Examples
    --------
    >>> CategoricalEncoder({"A": ['a', 'b', 'c']}).fit_transform(
    ...     pd.DataFrame({"A": ['a', 'b', 'a', 'a']})
    ... )['A']
    0    a
    1    b
    2    a
    3    a
    Name: A, dtype: category
    Categories (2, object): [a, b, c]
    """
    def __init__(self, categories):
        self.categories = categories

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col, categories in self.categories.items():
            print(col, categories)
            X[col] = X[col].astype('category').cat.set_categories(categories)
        return X


class ModifiedLabelEncoder(LabelEncoder):

    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)


class CategoryGroupEncoder(BaseEstimator, TransformerMixin):
    """
    Input a series of categories, returns a series with the most frequent `max_groups` unchanged,
    and others as "(Others)".
    `max_group` equals to the (ceiling of) square-root of number of categories by default.
    """
    def __init__(self, max_groups=None, groups=None):
        self.max_groups = max_groups  # number of categoricals to be encoded as is. Any other categories will become (Others)
        self.groups = groups  # pre-define a list of groups from prior knowledge presents

    def fit(self, X, y=None):
        if self.groups is not None:
            self.max_groups = len(self.groups)
            return self

        if self.max_groups is None:
            self.max_groups = int(np.ceil(np.sqrt(len(X.unique()))))
        self.groups = X.value_counts(sort=True, ascending=False).nlargest(self.max_groups).index.tolist()
        return self

    def transform(self, X):
        return X.map(lambda g: g if g in self.groups else '(Others)')

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

'''
class DataFrameTransformer(object):
    """

    """
    def __init__(self, X, pipeline_map, y=None):
        """

        :param X: a Pandas DataFrame object.
        :param pipeline_map:
        :param y:
        """
        self.X = X
        self.y = y
        self.pipeline_map = pipeline_map

    def column_transformer(self, X, columns, transforms):
        """
        Perform fit-and-transform on `columns` of DataFrame `X` according to `transforms`,
        returns the transformed DataFrame and fitted pipeline.
        :param X:
        :param columns:
        :param transforms:
        :return:
        """
        # Special treatment required for `CategoricalEncoder` class,
        # which is to build a dictionary of unique items in each column.
        if any([isinstance(item, CategoricalEncoder) for item in transforms]):
            # build dictionary
            categories = dict()
            for cat_col in columns:
                # Ignore nan (we can impute them before encoding, or by default represent it as all zeros after encoding)
                cats = X[~X[cat_col].isnull()][cat_col].unique().tolist()
                categories.update({cat_col: cats})
            # Supply the original CategoricalEncoder with `categories` argument
            transforms = [CategoricalEncoder(categories) if isinstance(elem, CategoricalEncoder) else item for item in transforms]

        # Fit and transform
        X_sel = X[columns].copy()
        pipe = make_pipeline(
            *transforms
        )
        X_fit = pipe.fit_transform(X_sel)

        # In general fit_transform returns Numpy array. Convert to DatFrame in that case.
        if not isinstance(X_fit, pd.DataFrame):
            X_fit = pd.DataFrame(X_fit, columns=columns, index=X_fit.index)
        return X_fit, pipe

    def dataframe_transformer(self, X, pipeline_map):
        """
        Fit and transform a DataFrame `X` according to the pipeline settings specified in `pipeline_map`.
        :param X: a Pandas DataFrame object.
        :param pipeline_map: mapping of column names and corresponding transforms.
        Format is a list of 2-tuple, the 1st element is a list of column names and 2nd a list of Sklearn transforms.
        [([`column_name_1`, `column_name_2`, ...], [Transform_1, Transform_2, ...])]
        For example,
        [
            (['country'], [CategoricalEncoder(), FunctionTransformer(pd.get_dummies, validate=False)]),
            (['income', 'population'], [Imputer('mean'), StandardScaler()]),
            (['fixed_column'], [])  # pass an empty list to indicate no transform required for column(s)
        ]
        :return: transformed DataFrame `X_transformed`, and the corresponding fitted pipeline.
        """
        lst_X_out = []
        lst_pipes = []
        for pair in pipeline_map:
            X_fit, pipe_fit = self.column_transformer(pair[0], pair[1])
            lst_X_out.append(X_fit)
            lst_pipes
        X_out = pd.concat(lst_X_out, axis=0)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
'''
