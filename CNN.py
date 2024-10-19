import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

X=pd.read_csv('processed_data\\theta.csv')

min_max_scaler = StandardScaler()
X= min_max_scaler.fit_transform(X)
X=pd.DataFrame(X)

y=pd.read_csv('processed_data\\class.csv')
y['1']=y['1'].map({0:0,1:1,2:1})
# print(X.head())
# print(y['1'].unique())


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape)

X_train=X_train.values.reshape(-1,19,1)

X_test=X_test.values.reshape(-1,19,1)

print(X_train.shape)

# print(X_test.shape)

# print(y_train.shape)

import tensorflow as tf
from tensorflow import keras


from keras.models import Sequential
from keras.layers import LSTM,TimeDistributed,Dense,Embedding,Flatten,Dropout


model=Sequential()

model.add(LSTM(20,input_shape=[19,1],return_sequences=True))
model.add(LSTM(20,return_sequences=True))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2,activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

model.fit(X_train,y_train,epochs=30,validation_data=[X_test,y_test])

model.save('./LSTM.keras')