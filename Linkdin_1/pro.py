import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# CSV loaded
with open('/home/divakar/Desktop/file.csv') as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
    print(readCSV)

   # for row in readCSV:
    #    print(row)


# Imported into numpy arrays
    filename = '/home/divakar/Desktop/file.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x=list(reader)
    data = np.array(x)
    print(data.shape)

# Train data and test data
train=pd.read_csv('/home/divakar/Desktop/file.csv')
test=pd.read_csv('/home/divakar/Desktop/file1.csv')
train['Type']='Train' #Create a flag for Train and Test Data set
test['Type']='Test'
fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set
fullData.columns # This will show all the column names
fullData.head(10) # Show first 10 records of dataframe
fullData.describe() #You can look at summary of numerical fields by using describe() function

ID_col = ['Book-id']
target_col = ["Key Topics.Author"]
cat_cols = ['Key Topics','Author']
num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col))
other_col=['Type']                                        #Test and Train Data set identifier

num_cat_cols = num_cols+cat_cols # Combined numerical and Categorical variables

#Create a new variable for each variable having missing value with VariableName_NA
# and flag missing value with 1 and other with 0

for var in num_cat_cols:
    if fullData[var].isnull().any()==True:
        fullData[var+'_NA']=fullData[var].isnull()*1





#create label encoders for categorical features
for var in cat_cols:
 number = LabelEncoder()
 fullData[var] = number.fit_transform(fullData[var].astype('str'))



train=fullData[fullData['Type']=='Train']
test=fullData[fullData['Type']=='Test']

features=list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col))

# Filter missing values
filename = handel_missing_values(filename, HEADERS[6], '?')
train_x, test_x, train_y, test_y = split_filename(filename, 0.7, Key Topic,[1:-1], Author[-1])

print("Train_x Shape :: "), train_x.shape
print("Train_y Shape :: "), train_y.shape
print("Test_x Shape :: "), test_x.shape
print("Test_y Shape :: "), test_y.shape

random.seed(100)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)

