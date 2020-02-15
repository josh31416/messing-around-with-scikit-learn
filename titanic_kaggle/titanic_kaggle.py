__output = {}
# SHOW_PLTS = True
SHOW_PLTS = False

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load the dataset
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

######################################################################
# Exploring the dataset
######################################################################


def descriptive_statistics(df, var_name="var"):
    __output[var_name] = {}
    __output[var_name] = __output[var_name]
    __output[var_name]["head"] = df.head()
    __output[var_name]["shape"] = df.shape
    __output[var_name]["describe_num"] = df.describe()
    __output[var_name]["describe_cat"] = df.describe(include="O")
#    __output[var_name]["info"] = df.info()
    __output[var_name]["is_null"] = df.isnull().sum()


descriptive_statistics(train, "_train")
descriptive_statistics(test, "_test")

######################################################################
# Relationship between features and "Survival"
######################################################################
survived = train[train["Survived"] == 1]
notSurvived = train[train["Survived"] == 0]

# Survived Ratio
survivedRatio, notSurvivedRatio = (len(survived)/len(train)*100, len(notSurvived)/len(train)*100)
print(f"Survived: {survivedRatio} %\n Not survived: {notSurvivedRatio} %")

# Pclass - Survived
__output["_train"]["Pclass-Survived"] = train[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean()
sns.barplot(x="Pclass", y="Survived", data=train)
if SHOW_PLTS: plt.show()

# Sex - Survived
__output["_train"]["Sex-Survived"] = train[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean()
sns.barplot(x='Sex', y='Survived', data=train)
if SHOW_PLTS: plt.show()

# Pclass & Sex - Survived
sns.factorplot("Sex", "Survived", hue="Pclass", size=4, aspect=2, data=train)
if SHOW_PLTS: plt.show()

# Pclass, Sex & Embarked - Survived
sns.factorplot(x="Pclass", y="Survived", hue="Sex", col="Embarked", data=train)
if SHOW_PLTS: plt.show()

# Embarked - Survived
__output["_train"]["Embarked-Survived"] = train[["Embarked", "Survived"]].groupby(["Embarked"], as_index=False).mean()
sns.barplot(x="Embarked", y="Survived", data=train)
if SHOW_PLTS: plt.show()

# Parch - Survived
__output["_train"]["Parch-Survived"] = train[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean()
sns.barplot(x="Parch", y="Survived", ci=None, data=train)
if SHOW_PLTS: plt.show()

# SibSp - Survived
train[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean()
sns.barplot(x="SibSp", y="Survived", ci=None, data=train)
if SHOW_PLTS: plt.show()

# Age - Survived
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True, ax=ax1)
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True, ax=ax2)
sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, ax=ax3)
if SHOW_PLTS: plt.show()

total_survived = train[train['Survived']==1]
total_not_survived = train[train['Survived']==0]
male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]
female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]
male_not_survived = train[(train['Survived']==0) & (train['Sex']=="male")]
female_not_survived = train[(train['Survived']==0) & (train['Sex']=="female")]

plt.figure(figsize=[15,5])
plt.subplot(111)
sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Age')
if SHOW_PLTS: plt.show()

plt.figure(figsize=[15,5])

plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Female Age')

plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Male Age')
if SHOW_PLTS: plt.show()

# Correlation
plt.figure(figsize=(15, 6))
sns.heatmap(train.drop("PassengerId", axis=1).corr(), vmax=0.6, square=True, annot=True)
if SHOW_PLTS: plt.show()

######################################################################
# Feature extraction
######################################################################

# Title feature
train_test_data = [train, test]

for dataset in train_test_data:
    dataset["Title"] = dataset.Name.str.extract(" ([A-Za-z]+)\.", expand=True)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Title - Survived
train[["Title", "Survived"]].groupby(["Title"], as_index=False).mean()

# Map Title
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Map Sex
for dataset in train_test_data:
    dataset["Sex"] = dataset["Sex"].map({"female": 1, "male": 0}).astype(int)

# NaN Embarked
print(train.Embarked.unique())
print(train.Embarked.value_counts())
for dataset in train_test_data:
    dataset["Embarked"] = dataset["Embarked"].fillna("S")

# Map Embarked
for dataset in train_test_data:
    dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

# NaN Age
for dataset in train_test_data:
    age_avg = dataset["Age"].mean()
    age_std = dataset["Age"].std()
    age_null_count = dataset["Age"].isnull().sum()
    age_null_random_list = np.random.randint(age_avg-age_std, age_avg+age_std, size=age_null_count)
    dataset["Age"][np.isnan(dataset["Age"])] = age_null_random_list
    dataset["Age"] = dataset["Age"].astype(int)

# AgeBand feature
train["AgeBand"] = pd.cut(train["Age"], 5)
print(train[["AgeBand", "Survived"]].groupby(["AgeBand"], as_index=False).mean())
for dataset in train_test_data:
    dataset.loc[ dataset["Age"] <= 16, "Age"] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

# NaN Fare
for dataset in train_test_data:
    dataset["Fare"] = dataset["Fare"].fillna(train["Fare"].median())

# FareBand feature
train['FareBand'] = pd.qcut(train['Fare'], 4)
print(train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

# FamilySize feature
for dataset in train_test_data:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
print(train[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index=False).mean())

######################################################################
# Feature selection
######################################################################

features_to_drop = ["Name", "SibSp", "Parch", "Ticket", "Cabin", "FamilySize"]
for dataset in train_test_data:
    dataset.drop(features_to_drop, axis=1, inplace=True)
train.drop(["PassengerId", "AgeBand", "FareBand"], axis=1, inplace=True)

######################################################################
# Classification
######################################################################

X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test = test.drop("PassengerId", axis=1)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

predictions = {}
accuracy = {}

# Logistic regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
predictions["LogisticRegression"] = clf.predict(X_test)
accuracy["LogisticRegression"] = round(clf.score(X_train, y_train) * 100, 2)

# SVM
clf = SVC()
clf.fit(X_train, y_train)
predictions["SVM"] = clf.predict(X_test)
accuracy["SVM"] = round(clf.score(X_train, y_train) * 100, 2)

# Linear SVM
clf = LinearSVC()
clf.fit(X_train, y_train)
predictions["LinearSVM"] = clf.predict(X_test)
accuracy["LinearSVM"] = round(clf.score(X_train, y_train) * 100, 2)

# KNN
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
predictions["KNN"] = clf.predict(X_test)
accuracy["KNN"] = round(clf.score(X_train, y_train) * 100, 2)

# Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions["DecisionTree"] = clf.predict(X_test)
accuracy["DecisionTree"] = round(clf.score(X_train, y_train) * 100, 2)

# Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
predictions["RandomForest"] = clf.predict(X_test)
accuracy["RandomForest"] = round(clf.score(X_train, y_train) * 100, 2)

# Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)
predictions["NaiveBayes"] = clf.predict(X_test)
accuracy["NaiveBayes"] = round(clf.score(X_train, y_train) * 100, 2)

# Perceptron
clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
predictions["NaiveBayes"] = clf.predict(X_test)
accuracy["NaiveBayes"] = round(clf.score(X_train, y_train) * 100, 2)

# SGD
clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
predictions["NaiveBayes"] = clf.predict(X_test)
accuracy["NaiveBayes"] = round(clf.score(X_train, y_train) * 100, 2)

######################################################################
# Confusion matrix
######################################################################

from sklearn.metrics import confusion_matrix
import itertools

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_random_forest)

class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix,
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent,
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)

pass
######################################################################
# Submission file to Kaggle
######################################################################

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions["RandomForest"]
})

submission.to_csv("dataset/submission.csv", index=False)
