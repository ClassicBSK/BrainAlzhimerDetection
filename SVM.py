import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score

X=pd.read_csv('processed_data\\alpha.csv')
X2=pd.read_csv('processed_data\\beta.csv')
X3=pd.read_csv('processed_data\\delta.csv')
X4=pd.read_csv('processed_data\\gamma.csv')
X5=pd.read_csv('processed_data\\theta.csv')

X=X3

# X=X.merge(X2,how="right",left_index=True,right_index=True)
# X=X.merge(X3,how="right",left_index=True,right_index=True)
# X=X.merge(X4,how="right",left_index=True,right_index=True)

X=X.merge(X5,how="right",left_index=True,right_index=True)


y=pd.read_csv('processed_data\\class.csv')

X=X.drop(X[y['1']==2].index)

y=y.drop(y[y['1']==2].index)

# y['1']=y['1'].map({1:0,2:1})

print(X.head())
# print(y['1'].unique())

# y=y.iloc[:-17]
# X=X.iloc[:-17]

print(f"{X.shape}-----{y.shape}")


svm_clf=Pipeline([
    ("scaler",StandardScaler()),
    ("linear_svc",LinearSVC(loss="hinge")),
])

svm_clf.fit(X,y)

y_train_pred=cross_val_predict(svm_clf,X,y,cv=3)

conf_matrix=confusion_matrix(y_train_pred,y)
accuracy=accuracy_score(y_true=y,y_pred=y_train_pred)
recall=recall_score(y_true=y,y_pred=y_train_pred)
f1_s=f1_score(y_true=y,y_pred=y_train_pred)
# print(conf_matrix)

print()
print(f"accuracy = {accuracy}")
print(f"recall = {recall}")
print(f"f1_score = {f1_s}")


