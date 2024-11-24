import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

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

# X=X.values.reshape(-1,19,5,1)

# print(X.head())
# print(y['1'].unique())
# print(X.shape)
# print(y.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# print(X_train.shape)

X_train=X_train.values.reshape(-1,19,5,1)

X_test=X_test.values.reshape(-1,19,5,1)

# print(X_train.shape)

# print(X_test.shape)

# print(y_train.shape)

import tensorflow as tf
from tensorflow import keras


from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,LSTM,Reshape

model= Sequential()

model.add(Conv2D(64,7,activation="relu",padding="same",input_shape=[19,5,1]))
model.add(MaxPooling2D(2,padding="same"))

model.add(Conv2D(128,3,activation="relu",padding="same"))
model.add(Conv2D(128,3,activation="relu",padding="same"))
model.add(MaxPooling2D(2,padding="same"))

model.add(Conv2D(256,3,activation="relu",padding="same"))
model.add(Conv2D(256,3,activation="relu",padding="same"))
model.add(MaxPooling2D(2,padding="same"))

model.add(Flatten())
model.add(Reshape((256,3)))
model.add(LSTM(20,input_shape=[512,3],return_sequences=True))
model.add(LSTM(20,return_sequences=True))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2,activation="softmax"))

print(model.summary())
checkpoint = ModelCheckpoint('LSTMCNNmodel.keras', 
    verbose=1, 
    monitor='val_accuracy',
    save_best_only=True, 
    mode='auto'
)  

model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
model.fit(X_train,y_train,epochs=300,callbacks=[checkpoint],validation_data=[X_test,y_test])

model.save('./LSTMCNN.keras')