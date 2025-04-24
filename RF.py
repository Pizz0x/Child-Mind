# Let's now try to find a better model, we start by focusing on random forest. Indeed bagging can look like a good starting point but to improve it we should make models independent
# With Random Forest, bagging is exploited to improve accuracy of base decision trees and each node is built on a small subset of the feature set to forces the algorithm to use different features than a basic decision tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

train_features = df_train.columns.tolist()
test_features = df_test.columns.tolist()
features_toremove =  list(set(train_features) - set(test_features) - {'sii'})

#####################
#### DATA PREPARATION
#####################

del df_train['id']
for col in features_toremove:
    del df_train[col]
df_train.dropna(subset=['sii'], inplace=True)

physical_measures_df = pd.read_csv('data/physical_measures.csv')
df_train = df_train.merge( physical_measures_df, on=['Basic_Demos-Age', 'Basic_Demos-Sex'], suffixes=('', '_avg')) # add the column of the average physical measures to each row in the dataframe

cols = ['Physical-BMI','Physical-Height','Physical-Weight','Physical-Waist_Circumference','Physical-Diastolic_BP','Physical-HeartRate','Physical-Systolic_BP']
for col in cols:
    df_train[col] = df_train[col].fillna(df_train[f"{col}_avg"])
    del df_train[f"{col}_avg"]


X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]

is_numerical = np.array([np.issubdtype(dtype, np.number) for dtype in X.dtypes])
numerical_idx = np.flatnonzero(is_numerical)
new_X = X.iloc[:, numerical_idx]

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_array = imputer.fit_transform(new_X)
new_X = pd.DataFrame(X_array, columns=new_X.columns, index=new_X.index)

categorical_idx = np.flatnonzero(is_numerical==False)
categorical_X = X.iloc[:, categorical_idx]
imputer = SimpleImputer(strategy='most_frequent')
X_array = imputer.fit_transform(categorical_X)
categorical_X = pd.DataFrame(X_array, columns=categorical_X.columns, index=categorical_X.index)


from sklearn.preprocessing import OneHotEncoder

oh = OneHotEncoder(sparse_output=False)
oh.fit(categorical_X)

encoded = oh.transform(categorical_X)
for i, col in enumerate(oh.get_feature_names_out()):
    new_X = new_X.copy()
    new_X[col] = encoded[:, i]


feature_names = new_X.columns.tolist()


X_train, X_test, y_train, y_test = train_test_split( new_X, y, test_size=0.20, random_state=42)
baseline_accuracy = y_train.value_counts().max() / y_train.value_counts().sum()
print (f"Majority class accuracy: {baseline_accuracy:.3f}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(class_weight='balanced')
parameters = { 'n_estimators': [50, 100],
    'max_leaf_nodes': [2, 5, 10, 30],
    'criterion': ['gini', 'entropy']
    }
tuned_model = GridSearchCV(model, parameters, cv=5, verbose=0)
tuned_model.fit(X_train, y_train)

print("Feature Importances:")
print(tuned_model.best_estimator_.feature_importances_)

print ("Best Score: {:.3f}".format(tuned_model.best_score_) )
print ("Best Params: ", tuned_model.best_params_)

test_acc = accuracy_score(y_true = y_test, y_pred = tuned_model.predict(X_test) )
print ("Test Accuracy: {:.3f}".format(test_acc) )
# basically a little better than the naive classifier

from sklearn.metrics import ConfusionMatrixDisplay


ConfusionMatrixDisplay.from_estimator(
    estimator=tuned_model.best_estimator_,
    X=X_test, y=y_test,
    cmap = 'Blues_r')

plt.show()
# with the random forest we obtain a better model, but still with an accuracy that is very similiar to the majority class accuracy that is our baseline
a = 1