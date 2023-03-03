import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def ANNdiabetes(array) :
    data = pd.read_csv("data/malaria.csv")
    X = data.iloc[:,:4].values
    Y = data.iloc[:,[5]].values
    LE1 = LabelEncoder()
    X[:,2] = np.array(LE1.fit_transform(X[:,2]))
    ct =ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder="passthrough")
    X = np.array(ct.fit_transform(X))
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
    ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
    ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
    ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
    result=ann.predict(sc.transform([array]))
    return result
