import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

import keras 

X=pd.read_csv('processed_data\\alpha.csv')
X2=pd.read_csv('processed_data\\beta.csv')
X3=pd.read_csv('processed_data\\delta.csv')
X4=pd.read_csv('processed_data\\gamma.csv')
X5=pd.read_csv('processed_data\\theta.csv')

# X=X3

X=X.merge(X2,how="right",left_index=True,right_index=True)
X=X.merge(X3,how="right",left_index=True,right_index=True)
X=X.merge(X4,how="right",left_index=True,right_index=True)

X=X.merge(X5,how="right",left_index=True,right_index=True)


y=pd.read_csv('processed_data\\class.csv')

X=X.drop(X[y['1']==2].index)

y=y.drop(y[y['1']==2].index)
min_max_scaler = StandardScaler()
X= min_max_scaler.fit_transform(X)
X=pd.DataFrame(X)

X=X.values.reshape(-1,19,5,1)

# print(X.head())
# print(y['1'].unique())
# print(X.shape)
# print(y.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# print(X_train.shape)

X_train=X_train.reshape(-1,19,5,1)

X_test=X_test.reshape(-1,19,5,1)

model=keras.models.load_model('./lstmcnnfinal.keras')

accuracy=model.evaluate(X_test,y_test,verbose=0)
print(f"accuracy = {accuracy[1]}")