import numpy as np
import pandas as pd
import  math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm, ensemble
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle

style.use('ggplot')

data_set_2009 = pd.read_csv('hackathon_dataset_2009.dat', sep=",", header=None)
data_set_2010 = pd.read_csv('hackathon_dataset_2010.dat', sep=",", header=None)
data_set_2011 = pd.read_csv('hackathon_dataset_2011.dat', sep=",", header=None)
data_set_result = pd.read_csv('hackathon_result.dat', sep=",", header=None)

result_cpy = pd.read_csv('hackathon_result.dat', sep=",",header=None)

data_set_2009.drop(data_set_2009.index[0], inplace=True)
data_set_2010.drop(data_set_2010.index[0], inplace=True)
data_set_2011.drop(data_set_2011.index[0], inplace=True)
data_set_result.drop(data_set_result.index[0], inplace=True)

#Set_a is for regression
data_set_2009_a = data_set_2009[[4,5,7]]
data_set_2010_a = data_set_2010[[4,5,7]]
data_set_2011_a = data_set_2011[[4,5,7]]
data_set_result_a = data_set_result[[4,5,7]]

#Set_b is for classification
data_set_2009_b = data_set_2009[[2,3,5,7]]
data_set_2010_b = data_set_2010[[2,3,5,7]]
data_set_2011_b = data_set_2011[[2,3,5,7]]
data_set_result_b = data_set_result[[2,3,5,7]]



frames_a = [data_set_2009_a, data_set_2010_a, data_set_2011_a]
df_all_a = pd.concat(frames_a)

frames_b = [data_set_2009_b, data_set_2010_b, data_set_2011_b]
df_all_b = pd.concat(frames_b)
# for i in range(1, len(data_set_2009_a)):
#     data_set_2009_a[6][i] = 1 if (data_set_2009_a[6][i] == 'Y') else 0

# for i in range(1, len(data_set_result_a)):
#     data_set_result_a[6][i] = 1 if (data_set_result_a[6][i] == 'Y') else 0



forecast_col = 5
# 5 is the index of quantity

df_all_a.fillna(0, inplace=True)#in machine learning you cnannot deal with null number
df_all_b.fillna(0, inplace=True)#in machine learning you cnannot deal with null number

data_set_result_a.fillna(0, inplace=True)#in machine learning you cnannot deal with null number
data_set_result_b.fillna(0, inplace=True)#in machine learning you cnannot deal with null number



df_all_a.dropna(inplace=True)
df_all_b.dropna(inplace=True)



X = np.array(df_all_a.drop([5],1))
y=np.array(df_all_a[5])
X = preprocessing.scale(X)
y=np.array(df_all_a[5])

X_lately = np.array(data_set_result_a.drop([5],1))


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = ensemble.RandomForestRegressor()

clf.fit(X_train,y_train)#fit corresponds to train
accuracy_a = clf.score(X_test, y_test)#score corresponds to test
# training data and score data cannot be the same coz score will overlap the training result, which makes no sense

forecast_set_a = clf.predict(X_lately)

print("The prediction_a info:")
print(forecast_set_a, accuracy_a)



for i in range(1, len(forecast_set_a)):
    result_cpy[5][i] = forecast_set_a[i]

result_cpy[5][33722] = 5
print(result_cpy)

result_cpy.to_csv(r'result.txt', header=None, index=None, sep=',', mode='a')



#The Following part is for classification
'''
X = np.array(df_all_b.drop([5],1))
y=np.array(df_all_b[5])
X = preprocessing.scale(X)
y=np.array(df_all_b[5])

# X=X[:-forecast_out]
#X_lately = X[-forecast_out:]
X_lately = np.array(data_set_result_b.drop([5],1))


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#clf=LinearRegression(n_jobs=-1)
#clf = svm.SVR()
#clf = ensemble.RandomForestRegressor()

clf.fit(X_train,y_train)#fit corresponds to train
accuracy_b = clf.score(X_test, y_test)#score corresponds to test
# training data and score data cannot be the same coz score will overlap the training result, which makes no sense

forecast_set_b = clf.predict(X_lately)

print("The prediction_b info:")
print(forecast_set_b, accuracy_b)
#print(forecast_set)
'''


