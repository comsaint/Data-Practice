"""
This script is to duplicate the result of 'gender_submission.csv', where we predict only by gender.
"""

import logging
from ..Loader import Loader
from ..Paths import DICT_PATHS

loader = Loader()
df_train = loader.read_original_data(table_code='train')
df_test = loader.read_original_data(table_code='test')


# Use gender column only: Sex=='male' implies dead (Survived=0)
# We do not need the training data actually...
df_test['Survived'] = df_test['Sex'] == 'female'
df_test['Survived'].replace({True: 1, False: 0}, inplace=True)

df_predictions = df_test[['PassengerId', 'Survived']]
df_predictions.to_csv(DICT_PATHS['predictions'], encoding='utf-8', index=False)

logging.info("Module completed successfully")
