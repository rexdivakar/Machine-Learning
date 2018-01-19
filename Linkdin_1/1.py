import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier

# CSV loaded
with open('/home/divakar/Desktop/Datsets/test/file.csv') as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
    print(readCSV)

   # for row in readCSV:
    #    print(row)


# Imported into numpy arrays
    filename = '/home/divakar/Desktop/Datsets/test/file.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x=list(reader)
    data = np.array(x)
    print(data.shape)

# Train data and test data
train=pd.read_csv('/home/divakar/Desktop/Datsets/test/file.csv')
test=pd.read_csv('/home/divakar/Desktop/Datsets/test/file1.csv')
train['Type']='Train'
test['Type']='Test'
fullData = pd.concat([train,test],axis=0)
fullData.columns
fullData.head(16)
fullData.describe()

ID_col = ['S/No']
target_col = ['Book-id']
cat_cols = ['Key Topics','Author']
num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col))
other_col=['Type']

#Test and Train Data set identifier


fullData.isnull().any()
num_cat_cols = num_cols+cat_cols
for var in num_cat_cols:
    if fullData[var].isnull().any()==True:
        fullData[var+'_NA']=fullData[var].isnull()*1

fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)



for var in cat_cols:
 number = LabelEncoder()
 fullData[var] = number.fit_transform(fullData[var].astype('str'))


 fullData["Key Topics"] = number.fit_transform(fullData["Key Topics"].astype('str'))

 train = fullData[fullData['Type'] == 'Train']
 test = fullData[fullData['Type'] == 'Test']

train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]

features=list(set(list(fullData.columns))-set(ID_col)-set(target_col)-set(other_col))

x_train = Train[list(features)].values
y_train = Train["Key Topics"].values
x_validate = Validate[list(features)].values
y_validate = Validate["Key Topics"].values
x_test=test[list(features)].values


random.seed(100)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)

#print(x_train,y_train)

status = rf.predict_proba(x_validate)
#print(status)

final_status = rf.predict_proba(x_test)
test["Key Topic","Author","Book-id"]=final_status[:,1]


#Test system

test.to_csv('/home/divakar/Desktop/Datsets/test/testfinal.csv',columns=['Author','Key Topic','Book-id'])

with open('/home/divakar/Desktop/Datsets/test/testfinal.csv') as csvfile:
    readfile = csv.reader(csvfile,delimiter=',')
    print(readfile)

