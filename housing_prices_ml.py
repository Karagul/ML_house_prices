import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
sns.set()

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
test_y = pd.read_csv('data/sample_submission.csv')

test = pd.merge(test, test_y, on='Id')

data = train.append(test, ignore_index=True)

# Dropping columns with a lot of missing values
# print(data.isna().sum().sort_values())
data = data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1)

# Dropping any rows with nulls
data = data[data.isna().sum(axis=1) == 0]

# Ordinal Categorical features
catf_ordered = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure'
                , 'KitchenQual', 'Functional', 'GarageQual', 'GarageCond']


for feature in catf_ordered:
    lbe = LabelEncoder()
    lbe = lbe.fit(list(data[feature].values))
    data[feature] = lbe.transform(list(data[feature].values))

data = pd.get_dummies(data)

n = round(len(data)*.75)

train = data.iloc[:n, :]
test = data.iloc[n:,:]

col_names=data.columns

scaler = StandardScaler()
train = scaler.fit_transform(X=train)
test = scaler.transform(X=test)

train = pd.DataFrame(train,columns=col_names)
test = pd.DataFrame(test, columns=col_names)
train_x = train.drop(['Id', 'SalePrice'],axis=1)
train_y = train['SalePrice']

test_x = test.drop(['Id', 'SalePrice'],axis=1)
test_y = test['SalePrice']

lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True, tol=.05)
lassocv.fit(train_x, train_y)
las = Lasso(alpha=lassocv.alpha_)
las.fit(train_x, train_y)

print(mean_squared_error(test_y, las.predict(test_x)))

plt.show()
