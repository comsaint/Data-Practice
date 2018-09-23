"""
Load parsed data sets (train and test)
"""
from ..Loader import Loader
from ..Paths import FULLPATH_DATA_TRAIN_PARSED, FULLPATH_DATA_TEST_PARSED

loader = Loader()
df_train, df_test = loader.run(mode='parsed')
df_train.to_csv(path_or_buf=FULLPATH_DATA_TRAIN_PARSED, encoding='utf-8', index=False)
df_test.to_csv(path_or_buf=FULLPATH_DATA_TEST_PARSED, encoding='utf-8', index=False)

print(df_train.describe())
print(df_test.describe())