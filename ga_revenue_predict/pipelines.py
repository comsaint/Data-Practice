"""
Pipelines and transforms to be performed.
"""
import logging
from .Loader import Loader
from .Settings import USE_COLS, NUM_COLS, TARGET
from .CommonFunctions import store_df_and_dtypes, deduplicate_repeated_sessionId
from sklearn_pandas import DataFrameMapper
from .sk_util import RemoveColumns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from .Paths import FULLPATH_DATA_TRAIN_PARSED, FULLPATH_DATA_TEST_PARSED


def preprocessing():
    # Preprocessing steps, dropping unuse columns, fix dtypes, merge duplicated session IDs, impute missing values.
    from numpy import log1p
    # Load raw data
    loader = Loader()
    #df_train, df_test = loader.run(mode='raw', subsample=0.01)
    df_train, df_test = loader.run(mode='raw')

    # De-duplicate
    df_train = deduplicate_repeated_sessionId(df_train)
    df_test = deduplicate_repeated_sessionId(df_test)

    # remove unused columns
    rc = RemoveColumns(USE_COLS)
    df_train_rc = rc.transform(df_train)
    df_test_rc = rc.transform(df_test)

    train_data_mapper = DataFrameMapper([
        (NUM_COLS, SimpleImputer(strategy='constant', fill_value=0.0)),
        (TARGET, [SimpleImputer(strategy='constant', fill_value=0.0), FunctionTransformer(log1p)])
    ], input_df=True, df_out=True, default=None)

    test_data_mapper = DataFrameMapper([
        (NUM_COLS, SimpleImputer(strategy='constant', fill_value=0.0)),
    ], input_df=True, df_out=True, default=None)

    df_train_parsed = train_data_mapper.fit_transform(df_train_rc)
    df_test_parsed = test_data_mapper.fit_transform(df_test_rc)
    # print(train_data_mapper.transformed_names_)

    df_train_parsed.rename({'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_0': 'totals.bounces',
                            'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_1': 'totals.hits',
                            'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_2': 'totals.newVisits',
                            'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_3': 'totals.pageviews'},
                           axis=1, inplace=True)
    df_test_parsed.rename({'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_0': 'totals.bounces',
                           'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_1': 'totals.hits',
                           'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_2': 'totals.newVisits',
                           'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_3': 'totals.pageviews'},
                          axis=1, inplace=True)
    df_train_parsed = df_train_parsed.astype(df_train_rc.dtypes.to_dict())
    df_test_parsed = df_test_parsed.astype(df_test_rc.dtypes.to_dict())

    df_train_parsed = df_train_parsed.astype({'totals.bounces': int,
                                              'totals.hits': int,
                                              'totals.newVisits': int,
                                              'totals.pageviews': int,
                                              'totals.transactionRevenue': float,
                                              'trafficSource.isTrueDirect': bool})
    df_test_parsed = df_test_parsed.astype({'totals.bounces': int,
                                            'totals.hits': int,
                                            'totals.newVisits': int,
                                            'totals.pageviews': int,
                                            'trafficSource.isTrueDirect': bool})

    print(df_train_parsed.head(10))
    print(df_train_parsed.dtypes)

    store_df_and_dtypes(df_train_parsed, path=FULLPATH_DATA_TRAIN_PARSED, index=False)
    store_df_and_dtypes(df_test_parsed, path=FULLPATH_DATA_TEST_PARSED, index=False)
    return 0
