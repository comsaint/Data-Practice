from .Loader import Loader
from .pipelines import RemoveColumns
from .Settings import USE_COLS, NUM_COLS, TARGET
from .Schemas import SCHEMAS
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from numpy import log1p
import pandas as pd
from .CommonFunctions import store_df_and_dtypes
from .Paths import FULLPATH_DATA_TRAIN_PARSED, FULLPATH_DATA_TEST_PARSED


# Load raw data
loader = Loader()
df_train, df_test = loader.run(mode='raw', subsample=0.01)

# remove unused columns
rc = RemoveColumns(USE_COLS)
df_train_rc = rc.transform(df_train)
df_test_rc = rc.transform(df_test)
print(df_train_rc.head())
print(df_train_rc.dtypes)

'''
column_trans_train = make_column_transformer(
    (NUM_COLS, SimpleImputer(strategy='constant', fill_value=0.0)),
    (TARGET, make_pipeline(SimpleImputer(strategy='constant', fill_value=0.0), FunctionTransformer(log1p))),
    remainder='passthrough', n_jobs=-1
)

column_trans_test = make_column_transformer(
    (NUM_COLS, SimpleImputer(strategy='constant', fill_value=0.0)),
    remainder='passthrough', n_jobs=-1
)

# Column order will be sorted after column transform, need to preserve
col_train_passthrough = NUM_COLS + TARGET + [col for col in df_train_rc.columns if col not in NUM_COLS+TARGET]


df_train_parsed = pd.DataFrame(column_trans_train.fit_transform(df_train_rc),
                               columns=col_train_passthrough)

for k, v in SCHEMAS['train_parsed'].items():
    df_train_parsed[k] = df_train_parsed[k].astype(v)

df_test_parsed = pd.DataFrame(column_trans_test.fit_transform(df_test_rc))

print(df_train_parsed.head())
print(df_train_parsed.dtypes)

store_df_and_dtypes(df_train_parsed, path=FULLPATH_DATA_TRAIN_PARSED, index=False)
store_df_and_dtypes(df_test_parsed, path=FULLPATH_DATA_TEST_PARSED, index=False)
'''
print(NUM_COLS)
from sklearn_pandas import DataFrameMapper
train_data_mapper = DataFrameMapper([
     (NUM_COLS, SimpleImputer(strategy='constant', fill_value=0.0)),
     (TARGET, [SimpleImputer(strategy='constant', fill_value=0.0), FunctionTransformer(log1p)])
], input_df=True, df_out=True, default=None)

test_data_mapper = DataFrameMapper([
     (NUM_COLS, SimpleImputer(strategy='constant', fill_value=0.0)),
], input_df=True, df_out=True, default=None)

df_train_parsed = train_data_mapper.fit_transform(df_train_rc)
df_test_parsed = test_data_mapper.fit_transform(df_test_rc)
#print(train_data_mapper.transformed_names_)

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
