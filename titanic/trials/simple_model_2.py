"""
Rewrite simple_model_1.py using sklearn-pandas. No CV search.
"""
from sklearn_pandas import DataFrameMapper, CategoricalImputer
import logging
from ..Loader import Loader
from ..Paths import DICT_PATHS
import pandas as pd
from sklearn.pipeline import make_pipeline
from ..sk_util import CategoricalEncoder
from sklearn.preprocessing import Imputer, FunctionTransformer, StandardScaler
from sklearn.linear_model import RidgeClassifierCV

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn_pandas import gen_features

loader = Loader()
df_train = loader.read_original_data(table_code='train')
df_test = loader.read_original_data(table_code='test')

# Consider only a subset of columns
df_train.set_index('PassengerId', inplace=True)
df_test.set_index('PassengerId', inplace=True)
#print(df_train.head())

USE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TARGET = ['Survived']
X_train = df_train[USE_COLS].copy()
y_train = df_train[TARGET].copy().values.reshape(-1,)
X_test = df_test[USE_COLS].copy()

# Preprocessing
# 1. 1-hot encode categorical columns
feature_cat = gen_features(columns=['Pclass', 'Sex', 'Embarked'],
                           classes=[CategoricalImputer, {'class': FunctionTransformer,
                                                         'func': pd.get_dummies,
                                                         'validate':False}]
                           )
feature_num = gen_features(columns=[['Age'], ['SibSp'], ['Parch'], ['Fare']],
                           classes=[Imputer, StandardScaler])
'''
mapper = DataFrameMapper([
    ('Sex', [CategoricalImputer(), FunctionTransformer(pd.get_dummies, validate=False)]),
    ('Embarked', [CategoricalImputer(), FunctionTransformer(pd.get_dummies, validate=False)]),
    (['Age', 'SibSp', 'Parch', 'Fare'], [Imputer(), StandardScaler()])
], df_out=True)
'''
mapper = DataFrameMapper(
    feature_cat + feature_num,
    input_df=True, df_out=True)

X_train_fit = mapper.fit_transform(X_train.copy())
#print(X_train_fit.head())
#print(X_train_fit.columns)

# Training
pipe_model = make_pipeline(
    RidgeClassifierCV()  # <- try different classifiers here!
)
pipe_model.fit(X_train_fit, y_train)
print(pipe_model.score(X_train_fit, y_train))

# Apply on test set
X_test_txf = mapper.transform(X_test.copy())
y_test_predict = pd.DataFrame(pipe_model.predict(X_test_txf), index=X_test.index, columns=TARGET)

# Write prediction
y_test_predict.to_csv(DICT_PATHS['predictions'], encoding='utf-8', index=True)

'''
# Feature selection
from sklearn.feature_selection import SelectKBest, mutual_info_classif
kselector = SelectKBest(mutual_info_classif, k=3)
X_new = kselector.fit_transform(X_train_fit, y_train)

print(pd.DataFrame([kselector.scores_], columns=X_train_fit.columns))

mapper_fs = DataFrameMapper([
    (list(X_train_fit.columns), SelectKBest(mutual_info_classif, k=2))
    ], input_df=True, df_out=True
)
fs = mapper_fs.fit_transform(X_train_fit, y_train)

#print(fs.head())
'''
