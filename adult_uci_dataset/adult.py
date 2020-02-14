__output = {}
SHOW_PLTS = False
#SHOW_PLTS = True

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

adult_columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                 "marital-status", "occupation", "relationship", "race", "sex",
                 "capital-gain", "capital-loss", "hours-per-week",
                 "native-country", "income"]

train = pd.read_csv("dataset/adult.data", names=adult_columns, skipinitialspace=True,
                    na_values=["?"])
test = pd.read_csv("dataset/adult.test", names=adult_columns, skipinitialspace=True,
                    na_values = ["?"])

######################################################################
# Exploring the dataset
######################################################################


def descriptive_statistics(df, var_name="var"):
    try: __output[var_name]
    except KeyError: __output[var_name] = {}
    __output[var_name] = __output[var_name]
    __output[var_name]["head"] = df.head()
    __output[var_name]["shape"] = df.shape
    __output[var_name]["describe_num"] = df.describe()
    __output[var_name]["describe_num"] = __output[var_name]["describe_num"]\
        .reset_index().replace(
            {'25%': '25_percent', '50%': '50_percent', '75%': '75_percent'}
        ).set_index('index')
    __output[var_name]["describe_cat"] = df.describe(include="O")
#    __output[var_name]["info"] = df.info()
    __output[var_name]["is_null"] = df.isnull().sum()


descriptive_statistics(train, "train")
descriptive_statistics(test, "test")


def cat_feature_value_counts(df, var_name, feature_name):
    try: __output[var_name]
    except KeyError: __output[var_name] = {}
    try: __output[var_name]["value_counts"]
    except KeyError: __output[var_name]["value_counts"] = {}
    __output[var_name]["value_counts"][feature_name] = df[feature_name].value_counts()


adult_cat_columns= train.select_dtypes(object).columns
for cat_column in adult_cat_columns:
    cat_feature_value_counts(train, "train", cat_column)

train.hist(bins=100)
if SHOW_PLTS: plt.show()

from pandas.plotting import scatter_matrix

scatter_matrix(train)
if SHOW_PLTS: plt.show()

######################################################################
# Relationship between features and income
######################################################################

train["income"] = train["income"].map({">50K": 1, "<=50K": 0})
test["income"] = test["income"].map({">50K.": 1, "<=50K.": 0})

high_income = train[train["income"] == 1]
low_income = train[train["income"] == 0]

# Income ratio
__output["train"]["high_income_ratio"], __output["train"]["low_income_ratio"] = \
    (len(high_income)/len(train)*100,
     len(low_income)/len(train)*100)

# Sex - Income
__output["train"]["Sex-Income"] = train[["sex", "income"]]\
    .groupby(["sex"], as_index=False).mean()
sns.barplot(x="sex", y="income", data=train)
if SHOW_PLTS: plt.show()

# Workclass - Income
__output["train"]["Workclass-Income"] = train[["workclass", "income"]]\
    .groupby(["workclass"], as_index=False).mean()
sns.barplot(x="workclass", y="income", data=train)
if SHOW_PLTS: plt.show()

# Occupation - Income
__output["train"]["Occupation-Income"] = train[["occupation", "income"]]\
    .groupby(["occupation"], as_index=False).mean()
sns.barplot(x="occupation", y="income", data=train)
if SHOW_PLTS: plt.show()

# Native-country - Income
__output["train"]["Native-country"] = train[["native-country", "income"]]\
    .groupby(["native-country"], as_index=False).mean()
sns.barplot(x="native-country", y="income", data=train)
if SHOW_PLTS: plt.show()

# Age - Income
fig = plt.figure()
ax1 = fig.add_subplot(111)
sns.violinplot(x="sex", y="age", hue="income", data=train, split=True, ax=ax1)
if SHOW_PLTS: plt.show()

# Features correlation
sns.heatmap(train.corr(), vmax=0.6, square=True, annot=True)
if SHOW_PLTS: plt.show()

######################################################################
# Feature engineering
######################################################################

datasets = [train, test]

# NaN handling
for dataset in datasets:
    dataset.dropna(subset=["occupation"], inplace=True)
    dataset.dropna(subset=["workclass"], inplace=True)
    dataset.drop(["native-country"], axis=1, inplace=True)

# Bands
# train["AgeBand"] = pd.cut(train["age"], 5)
# print(train[["AgeBand", "income"]].groupby(["AgeBand"], as_index=False).mean())
for dataset in datasets:
    dataset.loc[dataset["age"] <= 31.6, "age"] = 0
    dataset.loc[(dataset["age"] > 31.6) & (dataset["age"] <= 46.2)] = 1
    dataset.loc[(dataset["age"] > 46.2) & (dataset["age"] <= 60.8)] = 2
    dataset.loc[(dataset["age"] > 60.8) & (dataset["age"] <= 75.4)] = 3
    dataset.loc[dataset["age"] > 75.4, "age"] = 4
    dataset["age"] = dataset["age"].astype(int)

# Mapping


def onehot_mapping(df, feature_name):
    feature_values = df[feature_name].unique()
    for feature_value in feature_values:
        df[feature_value] = (df[feature_name] == feature_value).astype(int)
    df.drop([feature_name], axis=1, inplace=True)

for dataset in datasets:
    for cat_feature in dataset.select_dtypes(object).columns:
        onehot_mapping(dataset, cat_feature)

######################################################################
# Classification
######################################################################

X_train = train.drop("income", axis=1).values
y_train = train["income"].values
X_test = test.drop("income", axis=1).values
y_test = test["income"].values

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.predict(X_test)
print(round(clf.score(X_train, y_train)*100, 2))
print(round(clf.score(X_test, y_test)*100, 2))


pass

