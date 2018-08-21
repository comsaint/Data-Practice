"""
Previously, we achieve ~0.77 accuracy in simple_model_5 without much effort.
From here, we put more efforts on data exploration and feature engineering.
Tasks:
1. A better imputer for NAs
2. Generate more features by combining existing ones and crafting new ones
"""
import seaborn as sns
import matplotlib.pyplot as plt
from ..Loader import Loader

loader = Loader()
df_train = loader.read_original_data(table_code='train')
df_test = loader.read_original_data(table_code='test')

df_train.set_index('PassengerId', inplace=True)
df_test.set_index('PassengerId', inplace=True)

#print(df_train.info())
# Data Exploration
# Let's do plotting!
sns.set_style("whitegrid")

# 1. Gender vs Survived
#sns.barplot(x="Sex", y="Survived", hue='Survived', data=df_train, estimator=len)

# 2. Pclass vs Survived (similar to above, but use countplot)
#sns.countplot(x="Pclass", hue='Survived', data=df_train)

# 3. Age vs Survived (note there are substantial NAs in age)
#sns.catplot(x="Survived", y="Age", data=df_train, kind="swarm", hue='Sex')

# 4. Fare amount vs Embarked
#sns.catplot(x="Embarked", y="Fare", kind="box", data=df_train)

# 5. SibSp and Parch
sns.catplot(x="SibSp", y="Age", data=df_train, kind='box')
#plt.show()
sns.catplot(x="Parch", y="Age", data=df_train, kind='box')
#plt.show()

# Enough plotting. Now back to business.
# 1. We need to address the problem of NAs. There are 3 columns with NAs: Age, Cabin and Embarked.
# For Embarked, since only 2 entries are missing, we can simply use an imputer with most-frequent strategy.
# Significant Ages are missing, so we will build some kind of model to predict them,
# either the exact age or age group depending on how we decide to use this feature.
# A lot of Cabin are missing, we will think about how to craft features out of it.
# And just in case, we shall put a default imputer for all columns in the pipeline.

