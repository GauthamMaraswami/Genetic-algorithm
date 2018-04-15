# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 08:25:58 2018

@author: TBS
"""
from sklearn.preprocessing import StandardScaler
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense
water =pandas.read_csv("waterset.csv")
predictors=["MG","PH","NITRATE","BICARBONATE"]
water1=water[predictors]
water2=water["Water Quality"]
labelencoder_y_1 = LabelEncoder()
water2 = labelencoder_y_1.fit_transform(water2)
x_train=water1.iloc[0:15]
y_train=water2[0:15]
x_test=water1.iloc[15:]
y_test=water2[15:]

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
print(y_train)
print(x_test)
print(y_test)