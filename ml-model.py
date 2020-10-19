# FIFA Player Market Value Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')

to_stay=[
    "ID","Name","Age","Nationality","Club","Overall","Potential",
    "Value","Wage","Real Face"
]

dataset.drop(dataset.columns.difference(to_stay),axis="columns",inplace=True)


dataset.set_index("ID",inplace=True)
dataset.isnull().sum()


# Convert Value column from string to number
dataset['Value2'] = dataset['Value'].apply(lambda x: x.split('€')[1])
dataset['Value3'] = dataset['Value2'].apply(
    lambda x: float(x.split('M')[0])*1000000 
    if x.split('M').__len__() > 1 else float(x.split('K')[0])*1000
)

dataset.drop(['Value2', 'Value'], axis="columns",inplace=True)
dataset.rename(columns={"Value3":"Value"}, inplace=True)


# Convert Wage column from string to number
dataset['Wage2'] = dataset['Wage'].apply(lambda x: x.split('€')[1])
dataset['Wage3'] = dataset['Wage2'].apply(
    lambda x: float(x.split('M')[0])*1000000 
    if x.split('M').__len__() > 1 else float(x.split('K')[0])*1000
)

dataset.drop(['Wage2', 'Wage'], axis="columns",inplace=True)
dataset.rename(columns={"Wage3":"Wage"}, inplace=True)


# Checking for outliers in Age
import seaborn as sns
sns.set()
sns.boxplot(dataset['Age'])

q1 = np.percentile(dataset['Age'], 25)
q3 = np.percentile(dataset['Age'], 75)
iqr = q3 - q1
lower = q1 - (1.5 * iqr)
upper = q3 + (1.5 * iqr)

dataset['Age'][(dataset['Age'] < np.abs(lower)) | (dataset['Age'] > upper)].max()



X = dataset[['Age', 'Overall', 'Potential', 'Wage']]
y = dataset['Value']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

models_performance = []

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
l_regressor = LinearRegression()
l_regressor.fit(X_train, y_train)

# Predicting the Multiple Linear Regression model Test set results
y_pred = l_regressor.predict(X_test)

models_performance.append([
    'Linear Regression',
    r2_score(y_test, y_pred),
    mean_squared_error(y_test, y_pred),
    mean_absolute_error(y_test, y_pred),
])


# Training the Decision Tree Regression model on the Training set
from sklearn.tree import DecisionTreeRegressor
d_regressor = DecisionTreeRegressor(random_state = 0)
d_regressor.fit(X_train, y_train)

# Predicting the Decision Tree Regression model Test set results
y_pred = d_regressor.predict(X_test)

models_performance.append([
    'Decision Tree Regression',
    r2_score(y_test, y_pred),
    mean_squared_error(y_test, y_pred),
    mean_absolute_error(y_test, y_pred),
])


# Training the Random Forest Regression model on the Training set
from sklearn.ensemble import RandomForestRegressor
f_regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
f_regressor.fit(X_train, y_train)

# Predicting the Random Forest Regression model Test set results
y_pred = f_regressor.predict(X_test)

models_performance.append([
    'Random Forest Regression',
    r2_score(y_test, y_pred),
    mean_squared_error(y_test, y_pred),
    mean_absolute_error(y_test, y_pred),
])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
scaled_X_train = sc_X.fit_transform(X_train)
scaled_y_train = sc_y.fit_transform(y_train.values.reshape(len(y_train),1))

# Training the SVR model on the Training set
from sklearn.svm import SVR
s_regressor = SVR(kernel = 'rbf')
s_regressor.fit(scaled_X_train, scaled_y_train)

# Predicting the SVR model Test set results
y_pred = sc_y.inverse_transform(s_regressor.predict(sc_X.transform(X_test)))

models_performance.append([
    'SVR',
    r2_score(y_test, y_pred),
    mean_squared_error(y_test, y_pred),
    mean_absolute_error(y_test, y_pred),
])


# Training the Gradient Boosting Regression model on the Training set
from sklearn.ensemble import GradientBoostingRegressor
g_regressor = GradientBoostingRegressor(n_estimators = 500)
g_regressor.fit(X_train, y_train)

# Predicting the Gradient Boosting Regression model Test set results
y_pred = g_regressor.predict(X_test)

models_performance.append([
    'Gradient Boosting Regression',
    r2_score(y_test, y_pred),
    mean_squared_error(y_test, y_pred),
    mean_absolute_error(y_test, y_pred),
])

models_performance = pd.DataFrame(data = models_performance, columns = ['Model', 'R2 Score', 'MSE', 'MAE'])
