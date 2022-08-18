# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 13:08:49 2022

@author: Pratik
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.datasets import make_blobs

# 1. load the data
df_Pratiksinh  = pd.read_csv("C:/Users/user/Downloads/pima-indians-diabetes.csv")
#df_Pratiksinh

# 2. add the column names
df_Pratiksinh.columns =['preg','plas','pres','skin','test','mass','pedi','age','class']

# 3a. check the names and types of columns
print(df_Pratiksinh.dtypes)

df_Pratiksinh.head()

# 3b. check the missing values
print(df_Pratiksinh.isnull())
print("Count total NaN at each column in a DataFrame\n")
print(df_Pratiksinh.isnull().sum())

# 3c. Check the statistics of the numeric fields (mean, min, max, median, count,..etc.)
print(df_Pratiksinh.describe())

# 3d. Check the categorical values, if any.
for i in df_Pratiksinh.columns :
    print(df_Pratiksinh[i].dtype.name)

# 3f. Print out the total number of instances in each class 

# 4. Prepare a standard scaler transformer to transform all the numeric values

transformer_Pratiksinh = pd.DataFrame(StandardScaler().fit_transform(df_Pratiksinh),
                                 columns=df_Pratiksinh.columns, index=df_Pratiksinh.index)
transformer_Pratiksinh

# 5. Split the features from the class.
X=transformer_Pratiksinh.drop('class',axis=1)
print(X.shape)
y = transformer_Pratiksinh['class']
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
print(y.shape)

# 6. Split your data into train 70% train and 30% test, use 42 for the seed
random.seed(42)
X_train_Pratiksinh, X_test_Pratiksinh, y_train_Pratiksinh, y_test_Pratiksinh = train_test_split(X, y_transformed, test_size=0.3)

# 7. apply fit
svm = SVC()
svm.fit(X_train_Pratiksinh,y_train_Pratiksinh)


# exercise 1: Hard voting
# 8. 
log_clf_P = LogisticRegression(max_iter=1400)
rnd_clf_P = RandomForestClassifier()
svm_clf_P = SVC()
dec_clf_P = DecisionTreeClassifier(criterion="entropy", max_depth =42)
#use number of randomize decision trees on various sub samples using average to predict for more accuracy and control over fitting 
ext_clf_P = ExtraTreesClassifier()

# 9. Define a voting classifier that contains all the above classifiers as estimators, set the voting to hard
voting_clf_P = VotingClassifier(
    estimators=[('lr', log_clf_P), ('rf', rnd_clf_P), ('svc', svm_clf_P),('dt',dec_clf_P),('et',ext_clf_P)],
    voting='hard')

# 10. Fit the training data to the voting classifier and predict the first three instances of test data.
voting_clf_P.fit(X_train_Pratiksinh, y_train_Pratiksinh)

Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=8, random_state=1)
ynew = voting_clf_P.predict(Xnew)

# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

# 11. Print out for each classifier (including the voting classifier) and for each instance the predicted and the actual values

for clf in (log_clf_P, rnd_clf_P, svm_clf_P, dec_clf_P,ext_clf_P,voting_clf_P):
    clf.fit(X_train_Pratiksinh, y_train_Pratiksinh)
    Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=8, random_state=1)
    ynew = clf.predict(Xnew)
    print("\n")
    for i in range(len(Xnew)):
    	print("X=%s \nPredicted=%s" % (Xnew[i], ynew[i]))        
    y_pred = clf.predict(X_test_Pratiksinh)
    print("\naccuracy score")
    print(clf.__class__.__name__, accuracy_score(y_test_Pratiksinh, y_pred))

# exercise 2: Soft Voting

# 14.
log_clf_P = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf_P = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf_P = SVC(gamma="scale", probability=True, random_state=42)
dec_clf_P = DecisionTreeClassifier(criterion="entropy", max_depth =42)
ext_clf_P = ExtraTreesClassifier()

# Define a voting classifier that contains all the above classifiers as estimators, set the voting to soft
voting_clf_P = VotingClassifier(
    estimators=[('lr', log_clf_P), ('rf', rnd_clf_P), ('svc', svm_clf_P),('dt',dec_clf_P),('et',ext_clf_P)],
    voting='soft')

# Fit the training data to the voting classifier and predict the first three instances of test data.
voting_clf_P.fit(X_train_Pratiksinh, y_train_Pratiksinh)

Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=8, random_state=1)
ynew = voting_clf_P.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

# Print out for each classifier (including the voting classifier) and for each instance the predicted and the actual values

for clf in (log_clf_P, rnd_clf_P, svm_clf_P, dec_clf_P,ext_clf_P,voting_clf_P):
    clf.fit(X_train_Pratiksinh, y_train_Pratiksinh)
    Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=8, random_state=1)
    ynew = clf.predict(Xnew)
    print("\n")
    for i in range(len(Xnew)):
    	print("X=%s \nPredicted=%s" % (Xnew[i], ynew[i]))        
    y_pred = clf.predict(X_test_Pratiksinh)
    print("\naccuracy score")
    print(clf.__class__.__name__, accuracy_score(y_test_Pratiksinh, y_pred))

# exercise 3: Random forests & Extra Trees

# 15a.
#pipeline = make_pipeline(StandardScaler(), GaussianNB(priors=None))
pipeline1_Pratiksinh = Pipeline(steps=[('std',StandardScaler()),('ext', ext_clf_P)])

# 15b.
pipeline2_Pratiksinh = Pipeline(steps=[('std',StandardScaler()),('dec', dec_clf_P)])

# 16. Fit the original data to both pipelines
pipeline1_Pratiksinh.fit(X_train_Pratiksinh, y_train_Pratiksinh)
pipeline2_Pratiksinh.fit(X_train_Pratiksinh, y_train_Pratiksinh)

# 17. Carry out a 10 fold cross validation for both pipelines set shuffling to true and random_state to 42.
strat_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores1 = cross_val_score(pipeline1_Pratiksinh, X_train_Pratiksinh, y_train_Pratiksinh, cv=strat_k_fold)
scores2 = cross_val_score(pipeline2_Pratiksinh, X_train_Pratiksinh, y_train_Pratiksinh, cv=strat_k_fold)

# 18. Printout the mean score evaluation for both pipelines

print(pipeline1_Pratiksinh.score(X_train_Pratiksinh,y_train_Pratiksinh))
print(pipeline2_Pratiksinh.score(X_train_Pratiksinh,y_train_Pratiksinh))

# 19. Predict the test using both pipelines and printout the confusion matrix, precision, recall and accuracy scores 
y_pred1 = pipeline1_Pratiksinh.predict(X_test_Pratiksinh)
y_pred2 = pipeline2_Pratiksinh.predict(X_test_Pratiksinh)

# confusion matrix
print("For pipeline 1")
conf_matrix = confusion_matrix(y_true=y_test_Pratiksinh, y_pred=y_pred1)
print("Confusion matrix ")
print(conf_matrix)
print("Precision ")
print(precision_score(y_test_Pratiksinh, y_pred1))
print("Recall ")
print(recall_score(y_test_Pratiksinh, y_pred1))
print("Accuracy score ")
print(accuracy_score(y_test_Pratiksinh, y_pred1))

# confusion matrix
print("For pipeline 2")
conf_matrix = confusion_matrix(y_true=y_test_Pratiksinh, y_pred=y_pred2)
print("Confusion matrix ")
print(conf_matrix)
print("Precision ")
print(precision_score(y_test_Pratiksinh, y_pred2))
print("Recall ")
print(recall_score(y_test_Pratiksinh, y_pred2))
print("Accuracy score ")
print(accuracy_score(y_test_Pratiksinh, y_pred2))

# exercise 4: Extra Trees and Grid search

# 21 and 22. Fit your training data to the randomized gird search object
param_grid1 = {
    "ext__max_leaf_nodes": range(10,3000,20),
}
randomized1_63 = GridSearchCV(estimator=pipeline1_Pratiksinh,param_grid= param_grid1)
randomized1_63.fit(X_train_Pratiksinh, y_train_Pratiksinh)


param_grid2 = {
    "ext__max_depth": range(1,1000,2),
}
randomized2_63 = GridSearchCV(estimator=pipeline1_Pratiksinh,param_grid= param_grid2)
randomized2_63.fit(X_train_Pratiksinh, y_train_Pratiksinh)

# 23. Print out the best parameters and accuracy score for randomized grid search 
print("Best parameter for grid search 1 (CV score=%0.3f):" % randomized1_63.best_score_)
print(randomized1_63.best_params_)

print("Best parameter for grid search 2 (CV score=%0.3f):" % randomized2_63.best_score_)
print(randomized2_63.best_params_)

# 24. 

y_pred1 = randomized1_63.predict(X_test_Pratiksinh)
y_pred2 = randomized2_63.predict(X_test_Pratiksinh)

# 25. Printout the precision, re_call and accuracy
print("For grid search 1")
print("Precision ")
print(precision_score(y_test_Pratiksinh, y_pred1))
print("Recall ")
print(recall_score(y_test_Pratiksinh, y_pred1))
print("Accuracy score ")
print(accuracy_score(y_test_Pratiksinh, y_pred1))

print("\nFor grid search 2")
print("Precision ")
print(precision_score(y_test_Pratiksinh, y_pred2))
print("Recall ")
print(recall_score(y_test_Pratiksinh, y_pred2))
print("Accuracy score ")
print(accuracy_score(y_test_Pratiksinh, y_pred2))












