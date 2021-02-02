import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")
df = df.drop("Unnamed: 32", axis=1)
df.drop('id', axis=1, inplace=True)
l = list(df.columns)
features_mean = l[1:11]

features_se = l[11:21]

features_worst = l[21:]

df['diagnosis'].unique()
sns.countplot(df['diagnosis'], label="Count",);
df['diagnosis'].value_counts()

corr = df.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr);

df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
df['diagnosis'].unique()

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

lr_acc = accuracy_score(y_test, y_pred)
print(lr_acc)

results = pd.DataFrame()
tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

dtc_acc = accuracy_score(y_test, y_pred)
print(dtc_acc)

tempResults = pd.DataFrame({'Algorithm':['Decision tree Classifier Method'], 'Accuracy':[dtc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

rfc_acc = accuracy_score(y_test, y_pred)
print(rfc_acc)

tempResults = pd.DataFrame({'Algorithm':['Random Forest Classifier Method'], 'Accuracy':[rfc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]

from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)

tempResults = pd.DataFrame({'Algorithm':['Support Vector Classifier Method'], 'Accuracy':[svc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
