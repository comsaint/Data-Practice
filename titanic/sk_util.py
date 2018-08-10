"""
This module stores often-used Sklearn utilities, for example column selector, categorical encoder, time extractor,
 etc.
Reusing this module is encouraged.
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin


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
            X[col] = X[col].astype('category').cat.set_categories(categories)
        return X


class StandardScaler(TransformerMixin):
    """
    Scale a subset of the columns in a DataFrame
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.m = X[self.columns].mean()
        self.v = X[self.columns].std()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.columns] = X[self.columns].sub(self.m).div(self.v)
        return X
