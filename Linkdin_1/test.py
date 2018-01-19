import csv
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#  DATA UPLOADED

data=pd.read_csv("/home/divakar/Desktop/file.csv")
#print(data)
print(train.shape)

# TRAIN DATA UPLOADED

train=pd.read_csv("/home/divakar/Desktop/file1.csv")
print(train.shape)

# TEST DATA UPLOADED

test=pd.read_csv("/home/divakar/Desktop/11.csv")
print(test.shape)

# DATA FITTING
train['Type']='Train'
test['Type']='Test'
fullData = pd.concat([train,test],axis=0)
fullData.columns
fullData.head(16)
fullData.describe()
