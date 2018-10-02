"""
Pipelines and transforms to be performed.
"""
from sklearn.base import TransformerMixin

# Custom transformers for this project only


class RemoveColumns(TransformerMixin):
    """
    Drop unused columns from train and test set.
    """
    def transform(self, X):
        from Settings import USE_COLS
        if 'totals.transactionRevenue' not in X.columns:
            USE_COLS.remove('totals.transactionRevenue')
        return X[[USE_COLS]]
