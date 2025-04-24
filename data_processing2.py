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
model = DecisionTreeClassifier(class_weight='balanced') # give weight to the class that are inversely proportional to frequency
parameters = {'max_leaf_nodes': [30],
    'max_depth': [3, 5, 10, None],
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
# from the confusion matrix, we can see that the class 0 have most of the instances, so class 0 and 1 have a larger impact on the final measure, so we gave a weight inversely proportional to frequency
# in this way we have a lower accuracy but a model that can actually recognize the different classes