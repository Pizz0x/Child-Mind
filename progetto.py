import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

url_train = "data/train.csv"
df_train = pd.read_csv(url_train)

#####################
#### DATA PREPARATION
#####################

df_train.dropna(subset=['sii'], inplace=True) # remove rows where the value of sii is NaN
# we also remove a feature that doesn't give use any added value: id
del df_train['id']

# issues with data are k-nary categorical class label and missing values
# for the missing values we start with physical measures that depends on the sex and age of a child
# since physical measures depends on the age and the sex, that are not null values, i thought it would be better to insert the average measures for the specific age and sex from external sources, since it's a quite precise way to insert missing values, to do this i create a csv file with the average of the various physical values for each (age, sex) using a reliable source : the NHANES data
physical_measures_df = pd.read_csv('data/physical_measures.csv')
df_train = df_train.merge( physical_measures_df, on=['Basic_Demos-Age', 'Basic_Demos-Sex'], suffixes=('', '_avg')) # add the column of the average physical measures to each row in the dataframe

cols = ['Physical-BMI','Physical-Height','Physical-Weight','Physical-Waist_Circumference','Physical-Diastolic_BP','Physical-HeartRate','Physical-Systolic_BP']
for col in cols:
    df_train[col] = df_train[col].fillna(df_train[f"{col}_avg"])
    del df_train[f"{col}_avg"]

# Now that we have the complete values for physical measures, we can start working on the dataset to handle the other missing values and k-nary categorical class labels
# But first it's better to separate the feature of the classification task from the other one

X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]

# now what we want to do is replace the missing values of the various columns with the mean for the numerical values and with the mode for the categorical, so we first have to separate this two type of features:
is_numerical = np.array([np.issubdtype(dtype, np.number) for dtype in X.dtypes])  # array of boolean saying if a specific column is numeric or not
numerical_idx = np.flatnonzero(is_numerical) # we keep only columns that are numerical
new_X = X.iloc[:, numerical_idx]    # takes only the column that are numerical

#print(new_X.head(10))
# now that we have all the numerical column, we start to replace the NaN value of a specific column with the mean value of the column -> we do by using the SimpleImputer, later we can use a more complex model -> KNN Imputer: fills missing values based on nearest neighbors, in this way we take correlation into account. Iterative Imputer: treats each feature with missing values as a target and uses the rest as predictors

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_array = imputer.fit_transform(new_X)
new_X = pd.DataFrame(X_array, columns=new_X.columns, index=new_X.index)
#print(new_X.head(10))

# check if all there are still any nan value in the numerical features
# num_inst, num_features = new_X.shape
# for f in range(num_features):
#     col = new_X.iloc[:, f].astype(str)
#     print(f, np.unique(col))
# we can feel satisfied by this first part of the data processing

# now we still have to handle the categorical features, for them we have 2 things to do -> replace missing values with the mode, transform them with One-Hot Encoding
# one hot encoding for the remaining features, we can use it since we only have 4 possible values for each categorical feature : the seasons
categorical_idx = np.flatnonzero(is_numerical==False)
categorical_X = X.iloc[:, categorical_idx]
#print(categorical_X.head(10))
# we can see that there is some correlation between the various season features but as starter it's better to use the mode, then in case we can try applying a more complex model : iterative + encoding
imputer = SimpleImputer(strategy='most_frequent')
X_array = imputer.fit_transform(categorical_X)
categorical_X = pd.DataFrame(X_array, columns=categorical_X.columns, index=categorical_X.index)
# print(categorical_X.head(10))

# now that we have no more missing values, we can handle categorical labels using one-hot encoding
from sklearn.preprocessing import OneHotEncoder

oh = OneHotEncoder(sparse_output=False)
oh.fit(categorical_X)

#print(oh.categories_) # basically only seasons
encoded = oh.transform(categorical_X)
# print(oh.get_feature_names_out())
# we now add the encoded string features to the new data frame
for i, col in enumerate(oh.get_feature_names_out()):
    new_X = new_X.copy()
    new_X[col] = encoded[:, i]

#### END

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

# we tune the hyperparameter using the validation set -> automatic parameter tuning using Grid Search:
model = DecisionTreeClassifier()
parameters = {'max_leaf_nodes': [30]
    # 'max_depth': [3, 5, 10, None],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'criterion': ['gini', 'entropy']
    }
tuned_model = GridSearchCV(model, parameters, cv=5, verbose=0)
tuned_model.fit(X_train, y_train)
print("Feature Importances:")
print(tuned_model.best_estimator_.feature_importances_)

print ("Best Score: {:.3f}".format(tuned_model.best_score_) )
print ("Best Params: ", tuned_model.best_params_)

test_acc = accuracy_score(y_true = y_test, y_pred = tuned_model.predict(X_test) )
print ("Test Accuracy: {:.3f}".format(test_acc) )




# We can see that even with a simple decision tree classifier, we get a perfect classifier. 
# Unfortunately in the test set we have less features than in the training set, so to have a coherent predictor, I've decided to remove those features since in practice, they don't provide any help in the classification of the test set.
# # # we can see that the features that are not included in the t est set are the PCIAT features, that are 20-item scale that measures characteristics and behaviors associated with compulsive use of the Internet including compulsivity, escapism, and dependency
# # # so it's better to reprocess the cleaning without including the PCIAT features.

# # url = "data/test.csv"
# # df_test = pd.read_csv(url)
# # print(df_test.info())





# Data : Basic -> always a value, 