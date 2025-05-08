
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

from sklearn.impute import KNNImputer, SimpleImputer

# KNN Imputer: fills missing values based on nearest neighbors, in this way we take correlation into account. 
# Iterative Imputer: treats each feature with missing values as a target and uses the rest as predictors
imputer = SimpleImputer()
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

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Create the model for multiclass
model = XGBClassifier(
    objective='multi:softprob',  # Multiclass: probability distribution output
    num_class=4,                 # Number of classes (0,1,2,3 -> 4 classes)
    eval_metric='mlogloss',      # Multiclass log loss
    use_label_encoder=False,     # Avoid old warning
    n_jobs=-1,                   # Use all cores
    random_state=42,              # Reproducibility
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7
)
# selector = RFECV(base_model, step=3, cv=5, scoring='accuracy', n_jobs=-1)
# selector.fit(X_train, y_train)
# X_train_subset = X_train.iloc[:, selector.support_]
# X_test_subset = X_test.iloc[:, selector.support_]

model.fit(X_train, y_train)

# print("Selected Features: ", X_train.columns.tolist())
# print("Best Params: ", model.best_params_)
test_acc = accuracy_score(y_true = y_test, y_pred = model.predict(X_test) )
print("Test Accuracy: {:.3f}".format(test_acc) )
# basically a little better than the naive classifier

# print("Feature Importances:")
# print(model.best_estimator_.feature_importances_)
# subset_feature_names = X_train.columns.tolist()

# fig, ax = plt.subplots(figsize=(9, 4))
# ax.barh(range(X_train.shape[1]), sorted(tuned_model.best_estimator_.feature_importances_)[::-1])
# ax.set_title("Feature Importances")
# ax.set_yticks(range(X_train.shape[1]))
# ax.set_yticklabels(np.array(subset_feature_names)[np.argsort(tuned_model.best_estimator_.feature_importances_)[::-1]])
# ax.invert_yaxis() 
# ax.grid()

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    estimator=model,
    X=X_test, y=y_test,
    cmap = 'Blues_r')

plt.show()

# with the random forest we obtain a better model, but still with an accuracy that is very similiar to the majority class accuracy that is our baseline
