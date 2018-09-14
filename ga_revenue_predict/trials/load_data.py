"""
Load parsed data sets (train and test)
"""
from ..Loader import Loader
loader = Loader()
df_train, df_test = loader.run(mode='parsed')

print(df_train.describe())