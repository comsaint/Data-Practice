"""
Module which stores the default schemas of tables.
"""

SCHEMAS = {
    'train': {
        'PassengerId': str,
        'Survived': int,
        'Pclass': 'category',
        'Name': str,
        'Sex': 'category',
        'Age': float,
        'SibSp': float,
        'Parch': float,
        'Ticket': str,
        'Fare': float,
        'Cabin': str,
        'Embarked': 'category'
    },
    'test': {
        'PassengerId': str,
        'Pclass': 'category',
        'Name': str,
        'Sex': 'category',
        'Age': float,
        'SibSp': float,
        'Parch': float,
        'Ticket': str,
        'Fare': float,
        'Cabin': str,
        'Embarked': 'category'
    },
    'submission': {
        'PassengerId': str,
        'Survived': int
    }
}