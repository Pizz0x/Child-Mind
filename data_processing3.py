import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# We can see that even with a simple decision tree classifier, we get a perfect classifier. 
# Unfortunately in the test set we have less features than in the training set, so to have a coherent predictor, I've decided to remove those features since in practice, they don't provide any help in the classification of the test set.
# # # we can see that the features that are not included in the test set are the PCIAT features, that are 20-item scale that measures characteristics and behaviors associated with compulsive use of the Internet including compulsivity, escapism, and dependency
# # # so it's better to process the cleaning without including the PCIAT features:

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

train_features = df_train.columns.tolist()
test_features = df_test.columns.tolist()
features_toremove =  list(set(train_features) - set(test_features) - {'sii'})

#####################
#### DATA PREPARATION
#####################

# since in the last data processing test we got an accuracy that is basically at the level of the baseline accuracy, we now want to try removing the rows and features that have a huge number of NaN column since they don't give us relevant insight in the data and tend to normalize data too much.
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


values = df_train.isna().sum()
percentages = (df_train.isna().sum()/(len(df_train)/100)).round(1)
nan_data = pd.DataFrame({
    "number_of_nan" : values,
    "percentages" : percentages
})
print(nan_data)
# as we saw a lot of columns have an high percentage of NaN values, which can affect the accuracy of our predictor, so I've decided to drop them from our training set.
# I've also decided to try deleting also the rows with an high percentage of NaN values, let's see what happen:
max_nan_c = int(df_train.shape[0] * 0.5)
max_nan_r = int(df_train.shape[1] * 0.5)
df_train = df_train.dropna(axis=1, thresh=max_nan_c)
df_train = df_train.dropna(thresh=max_nan_r)
print(df_train.shape)


X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]

is_numerical = np.array([np.issubdtype(dtype, np.number) for dtype in X.dtypes])
numerical_idx = np.flatnonzero(is_numerical)
new_X = X.iloc[:, numerical_idx]

from sklearn.impute import KNNImputer, SimpleImputer

# KNN Imputer: fills missing values based on nearest neighbors, in this way we take correlation into account. 
# Iterative Imputer: treats each feature with missing values as a target and uses the rest as predictors
imputer = KNNImputer(n_neighbors=5)
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

# First Approach with a Decision Tree
X_train, X_test, y_train, y_test = train_test_split( new_X, y, test_size=0.20, random_state=42)
# print(y_train_80.value_counts())
baseline_accuracy = y_train.value_counts().max() / y_train.value_counts().sum()
print (f"Majority class accuracy: {baseline_accuracy:.3f}") # this is the accuracy of the naive classifier saying sii value is 0.0 all the time

# we further split the train set to simulate an unseen test set on which we can tune/validate the algorithm hyper-parameters, to do that we use k-fold cross-validation since also some specific values in the validation set may affect the model performance and hyper parameter choices

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


model = RandomForestClassifier()

# we tune the hyperparameter using the validation set -> automatic parameter tuning using Grid Search:
#model = DecisionTreeClassifier() # give weight to the class that are inversely proportional to frequency
parameters = { 'n_estimators': [50, 100],
    'max_leaf_nodes': [2, 5, 10, 30],
    'criterion': ['gini', 'entropy']
    }
tuned_model = GridSearchCV(model, parameters, cv=5, n_jobs=-1)
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
# from the confusion matrix, we can see that the class 0 have most of the instances, so class 0 and 1 have a larger impact on the final measure, so we gave a weight inversely proportional to frequency
# in this way we have a lower accuracy but a model that can actually recognize the different classes