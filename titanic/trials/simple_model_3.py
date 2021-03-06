"""
Same as simple_model_2, but use ensemble model.
"""
from sklearn_pandas import DataFrameMapper, CategoricalImputer
import logging
from ..Loader import Loader
from ..Paths import DICT_PATHS
import pandas as pd
from sklearn.pipeline import make_pipeline
from ..sk_util import CategoricalEncoder
from sklearn.preprocessing import Imputer, FunctionTransformer, StandardScaler

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn_pandas import gen_features

loader = Loader()
df_train = loader.read_original_data(table_code='train')
df_test = loader.read_original_data(table_code='test')

# Consider only a subset of columns
df_train.set_index('PassengerId', inplace=True)
df_test.set_index('PassengerId', inplace=True)

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
mapper = DataFrameMapper(
    feature_cat + feature_num,
    input_df=True, df_out=True)

X_train_fit = mapper.fit_transform(X_train.copy())

# Training
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
import scipy

# Hyperparameters
param_grid_GB = {
    'learning_rate': scipy.stats.uniform(loc=0.01, scale=0.15),
    'max_depth': scipy.stats.randint(low=2, high=X_train_fit.shape[1]),
    'min_samples_split': scipy.stats.uniform(loc=0.001, scale=0.02),
    'min_samples_leaf': scipy.stats.uniform(loc=0.01, scale=0.2),
    'subsample': scipy.stats.uniform(loc=0.5, scale=0.4),
    'max_features': ['auto', 'sqrt', 'log2', None],
}

param_grid_LR = {
    'penalty': ['l1', 'l2'],
    'C': scipy.stats.expon(scale=10)
}

param_grid_LRCV = {}

param_RF = {
    'n_estimators': scipy.stats.randint(5, 1000),
    'max_features': ['auto', 'sqrt'],
    'max_depth': scipy.stats.randint(10, 100),
    'min_samples_leaf': scipy.stats.randint(1, 4),
    'min_samples_split': scipy.stats.randint(2, 10),
    'bootstrap': [True, False]
}

params = param_RF

model = RandomForestClassifier()

clf = RandomizedSearchCV(model, param_distributions=params, cv=5, verbose=1, n_jobs=-1, n_iter=200)

logging.info("Training...")
best_model = clf.fit(X_train_fit, y_train)

# Print best results (rank by high test score and low std)
train_result = pd.DataFrame(clf.cv_results_)
train_result.sort_values(by=['mean_test_score', 'std_test_score'], ascending=[False, True], inplace=True)
print(train_result[['mean_test_score', 'std_test_score']].head())

for param in params:
    print('Parameter: {}, best value={}'.format(param, best_model.best_estimator_.get_params()[param]))
#print("Accuracy: {}".format(best_model.cv_results_['mean_test_score']))

'''
pipe_model = make_pipeline(
    RandomizedSearchCV(LogisticRegression(),
                       param_distributions=param_grid_LR,
                       cv=5)
)



pipe_model.fit(X_train_fit, y_train)
print(pipe_model.score(X_train_fit, y_train))
'''

# Apply on test set
X_test_txf = mapper.transform(X_test.copy())
logging.info("Predicting...")
y_test_predict = pd.DataFrame(best_model.predict(X_test_txf), index=X_test.index, columns=TARGET)

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
