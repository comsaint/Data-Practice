"""
Run once to parse the raw data to flat CSV.
"""
from ..Loader import Loader
from ..Paths import FULLPATH_DATA_TRAIN_PARSED, FULLPATH_DATA_TEST_PARSED
# Load data
loader = Loader()
df_train, df_test = loader.run(mode='raw')

# Write to file for later use
df_train.to_csv(path_or_buf=FULLPATH_DATA_TRAIN_PARSED, encoding='utf-8', index=False)
df_test.to_csv(path_or_buf=FULLPATH_DATA_TEST_PARSED, encoding='utf-8', index=False)

#print(df_train.head())
print(df_train.dtypes)
print(df_train.columns)
#print(df_test.head())
#print(df_test.dtypes)
