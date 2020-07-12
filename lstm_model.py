# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 20:51:20 2020

@author: Nidhi Rai
"""

## Air Quality prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno




import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_excel("C:\\Users\\abc\\Desktop\\AirPollution_Vinod\\Delhi (1).xlsx", na_values = '-')

df.shape

df.head()

df.tail()

df.describe()

df.isna().sum()

##### data should be present for 4 months*24hours
#january= 31
#feburary = 28
#march = 31
#April = 20

(31+28+31+19)*24+1

#80 PM2.5 values are missing

df.dtypes

# sorting data frame by date
df.sort_values("date", axis = 0, ascending = True, 
                 inplace = True, na_position ='last')

df.head()

df.tail()

df.dtypes

## creating dataframe to apply merge columns for 24 hrs of each day from 2018-01-01 00:00:00 to 2018-04-20 00:00:00

df1 = pd.DataFrame(
        {'date': pd.date_range('2018-01-01', '2018-12-31', freq='1H', closed='left')}
     )

df2 = df1.iloc[:2617,:]

df4 = df1.iloc[2617:,:]
df2.tail

df3 = pd.merge(df,df2,on='date',how='right')
 
df3.info()

df3.sort_values("date", axis = 0, ascending = True, 
                 inplace = True)

df3.head()

#checking the count of missing values

df3.isna().sum()

# Visualize missing values as a matrix 

msno.bar(df3, figsize= (5,5), color = '0.25')

sns.heatmap(df3.isnull(), cbar=True)
'''
## Data Visualization and Feature Engineering
1. We need to impute the missing data
2. Extracting days of the date, will check the trend day wise
3. will add a variable if there is holiday or not on the given day
4. then will chcek the trend for monthly trend, daily trend, hourly trend
5. we are going to use different methods for imputing methods: interpolation, Kalman, locf 
'''

df3.info

df3['Time'],df3['Date']= df3['date'].apply(lambda x:x.time()), df3['date'].apply(lambda x:x.date())

df3.info()

## mean pm2.5 level for daywise
##setting index as datetime column
df3.set_index(['date'], inplace = True)
df3.head()

df3.dtypes



backup_data = df3
df3.tail()


df3['Date'] = pd.to_datetime(df3["Date"])

## creating Days of Week field as per Date
df3['day_of_week'] = df3['Date'].dt.day_name()
df3.head()

backup_data.head()
backup_data.tail()

df3 = backup_data
df3.day_of_week.value_counts()

## pm2.5 values by week
weekly_pm25= df3.groupby("day_of_week").mean()
weekly_pm25.sort_values("pm25")

## plot pm2.5 level with rescpect to days
weekly_pm25.plot()

## with the graph we can see that the pm2.5 level is least on Sundays, followed by monday tuesday and Saturday

## Drawing box plot 
plt.figure(figsize=(7,5))
sns.barplot(x= 'day_of_week', y = 'pm25', data = df3)

## On fridays the pollution is expected to higher than other days
## However the Saturday is weekend, pm2.5 value is higher than expected due to higher value on Friday.

##creating month column

df3['Month'] = df3['Date'].dt.month_name()
print(df3)

## pm2.5 values by week
monthly_pm25= df3.groupby("Month").mean()
monthly_pm25.sort_values("pm25")

monthly_pm25.plot()

## Drawing box plot 
plt.figure(figsize=(7,5))
sns.boxplot(x= 'Month', y = 'pm25', data = df3)

## Visualize Outlier data in month of april


'''
###### as we can see monthly downward trend is present in the pm2.5 level, with a given set of data for 4 months
1.there are other factors which are effecting the pm2.5 level

2. With the research, it was found that crop burning in 

3.due to data access limitation, we could find the daily temperature value, and value for rest of the parameters on monthly basis
'''

plt.figure(figsize=(10,20))
sns.catplot(x = "pm25",y="Month", hue="day_of_week", kind = 'bar',data=df3, height= 7, aspect=2)
plt.show()

## Lets explore the data on hourly basis 

hourly_pm25 = df3.groupby("Time").mean()
hourly_pm25.sort_values("pm25")


## to make the data more readable lets bin the time
hours = pd.to_datetime(df3['Time'], format='%H:%M:%S').dt.hour

df3['Time_bin'] = pd.cut(hours, 
                    bins=[0,2,4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], 
                    include_lowest=True, 
                    labels=['00:00 - 02:00', '02:00 - 04:00', '04:00 - 06:00', '06:00 - 08:00','08:00 - 10:00','10:00 - 12:00',
                           '12:00 - 14:00', '14:00 - 16:00', '16:00 - 18:00', '18:00 - 20:00','20:00 - 22:00','22:00 - 00:00'])

Two_hourly_pm25 = df3.groupby("Time_bin").mean()
Two_hourly_pm25.sort_values("pm25")

## plot the values to visualize the trend
Two_hourly_pm25.plot(figsize = (10,5))

## Certainly there is seasonality present in the pm2.5 level on the hourly basis



## Boxplot for hourly change in pm2.5 level

fig, ax = plt.subplots(figsize=(20,10))

sns.barplot(x="Time_bin",y="pm25",data=df3, hue = 'Month')

#there is almost similar trend in all the months of pm2.5 value change with respect to time in Delhi Air
# there is considerable drop in pm2.5 level between 2:00pm to 8:00pm in the month of march
# there are more holidays in the month of march,  which could have impacted the lower pm2.5 level

## Visualizing hourly change of pm2.5 for each Month

def dateplot(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)


g = sns.FacetGrid(df3, col="Month", height=3.5, aspect= 2, col_wrap = 2)
g = g.map_dataframe(dateplot, "Time_bin", "pm25")

## The trend in pm2.5 level is differing with month


df3.info()

df3.shape

df3.head


# imputing using the interpolate method=time
df3_linear = df3['pm25'].interpolate(method= 'linear')
df3_linear.isnull().sum()

bckup_data = df3
print(bckup_data)


df3 = df3[['pm25']]
print(df3)



df3_FillMedian = df3.assign(FillMedian=df3.pm25.fillna(df3.pm25.median()))

# imputing using the rolling average
df3_RollingAverage = df3.assign(RollingMean=df3.pm25.fillna(df3.pm25.rolling(24,min_periods=1,).mean()))
# imputing using the rolling median
df3_RollingMedian = df3.assign(RollingMedian=df3.pm25.fillna(df3.pm25.rolling(24,min_periods=1,).median()))


#Imputing using interpolation with different methods



df3_InterpolateLinear= df3.pm25.interpolate(method='linear')
df3_InterpolateTime= df3.pm25.interpolate(method='time')
df3_InterpolateQuadratic= df3.pm25.interpolate(method='quadratic')
df3_InterpolateCubic= df3.pm25.interpolate(method='cubic')
df3_InterpolateSLinear= df3.pm25.interpolate(method='slinear')
df3_InterpolateAkima= df3.pm25.interpolate(method='akima')
df3_InterpolatePoly5= df3.pm25.interpolate(method='polynomial', order=5)
df3_InterpolatePoly7= df3.pm25.interpolate(method='polynomial', order=7)
df3_InterpolateSpline3= df3.pm25.interpolate(method='spline', order=3)
df3_InterpolateSpline4= df3.pm25.interpolate(method='spline', order=4)
df3_InterpolateSpline5= df3.pm25.interpolate(method='spline', order=5)


# moving average for the time series to understand better about the trend

plt.figure(figsize=(20,3))

df3.pm25.plot(label="org")
for i in range(0,25,12):
    df3["pm25"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)

df3_InterpolateLinear.plot(figsize=(15,3), color="#F8766D", title='DELHI AIR QUALITY')
plt.show()

df3_InterpolateSLinear.plot(figsize=(15,3), color="#F8766D", title='DELHI AIR QUALITY')
plt.show()

df3_InterpolateTime.plot(figsize=(15,3), color="#F8766D", title='DELHI AIR QUALITY')
plt.show()

rolmean = pd.Series(df3_InterpolateTime).rolling(window = 24).mean()
rolstd = pd.Series(df3_InterpolateTime).rolling(window = 24).std()

##plot rolling statistics
plt.figure(figsize=(15,3)) 
orig=plt.plot(df3_InterpolateTime,color='blue',label='original')
mean=plt.plot(rolmean,color='red',label='rolling mean')
std=plt.plot(rolstd,color='black',label='rolling std')
plt.title('rolling mean & standard deviation')

##It can be observed from rolling mean and rolling std deviation is not constant, Hence we will check the ADFuller test

plt.figure(figsize=(15,3))




plt.plot(df3['pm25'], label='Actual')
plt.plot(df3_InterpolateLinear, label='Linear')


plt.legend()
plt.show()

from statsmodels.tsa.stattools import adfuller
def test_stationarity(df3_InterpolateLinear):
    
#Determing rolling statistics
    rolmean = pd.Series(df3_InterpolateLinear).rolling(window = 24).mean()
    rolstd = pd.Series(df3_InterpolateLinear).rolling(window = 24).std()
    
    #plot rolling statistics
    plt.figure(figsize=(15,3)) 
    orig=plt.plot(df3_InterpolateLinear,color='blue',label='original')
    mean=plt.plot(rolmean,color='red',label='rolling mean')
    std=plt.plot(rolstd,color='black',label='rolling std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(df3_InterpolateLinear, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


test_stationarity(df3_InterpolateLinear)
## As the p-value is too low and T-statistic value is also lower than the critical values
## the ADF test has an alternate hypothesis of linear or difference stationary
## As per rolling stats we can see that data is not showing contant mean and std deviation

df3 = bckup_data[['pm25','Time', 'Month', 'day_of_week']]
df3.head

## assigning pm25 as the Linear interpolation values
df3['pm25'] = df3_InterpolateLinear
#df3.reset_index(inplace = True)
df3.head()


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df3_InterpolateLinear)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig = plt.figure(figsize=(20,8))
plt.subplot(411)
plt.plot(df3_InterpolateLinear, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

## Normalizing pm2.5 values to 0-1 scale, as pre-requisite for LSTM
## Diving data in train, test and testFinal, train and test will be part of model creation in LSTM.
## Final testing will be done on testFinal dataset, which will be unseen for training model.

from sklearn.preprocessing import MinMaxScaler
dataset = df3.pm25[:2092].values #numpy.ndarray

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = np.reshape(dataset, (-1, 1))

dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

## final testing dataset
test_final = df3.pm25[2092:].values

test_final = np.reshape(test_final, (-1, 1))
test_final = scaler.fit_transform(test_final)

## validating if all values in all the 3 datasets are in 0-1 range
train, test, test_final

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)



# reshape into X=t and Y=t+1
look_back = 24
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)
X_testFinal, Y_testFinal = create_dataset(test_final, look_back)


X_train.shape

Y_test.shape


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
X_testFinal = X_testFinal.reshape(X_testFinal.shape[0],X_testFinal.shape[1] , 1)

X_train.shape, X_testFinal.shape, X_test.shape


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

## training the model
model=Sequential()
model.add(LSTM(100,return_sequences=True,input_shape=(24,1)))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(100))

model.add(Dense(1))
#model.add(Dropout(0.8))
model.compile(loss='mean_squared_error',optimizer='adam')

history = model.fit(X_train, Y_train, epochs=500, batch_size=40, validation_data=(X_test, Y_test), 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)


# Training Phase
model.summary()

from scipy import stats
# make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
testFinal_predict = model.predict(X_testFinal)

## validating format of predicted train, test and testFinal values
train_predict, test_predict, testFinal_predict


Y_testFinal, Y_test, Y_train

# Transform the train and test set in its original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
testFinal = scaler.inverse_transform(testFinal_predict)

## Now check the train and test predicted values
test_predict, train_predict,testFinal

## Apply reverse transformations on Y_train and Y_test and check the values
Y_train = scaler.inverse_transform([Y_train])
Y_test = scaler.inverse_transform([Y_test])
Y_testFinal = scaler.inverse_transform([Y_testFinal])
Y_train, Y_test, Y_testFinal

## Compairing the values of Y_testFinal and testFinal predicted values
Y_testFinal, testFinal[:,0]

###### Visualizing the errors, loss values and graphs for actual and predicted values
#### ets check the error in train, test and testFinal sets


import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Validation Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Validation Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_testFinal[0], testFinal[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_testFinal[0], testFinal[:,0])))


## Lets plot the Loss Function for train and test set
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();

## Lets plot 394 observations of test set

aa=[x for x in range(394)]
plt.figure(figsize=(15,4))
plt.plot(aa, Y_test[0][:394], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:394], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('pm25', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();


## Let's plot train set for Actual and predicted values

aa=[x for x in range(1375)]
plt.figure(figsize=(15,4))
plt.plot(aa, Y_train[0][:1375], marker='.', label="actual")
plt.plot(aa, train_predict[:,0][:1375], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
#plt.subplots_adjust(left=0.07)
plt.ylabel('pm25')
plt.xlabel('Time step')
plt.legend(fontsize=15)
plt.show();


## Compairing the values of Y_test and test predicted values

pd.DataFrame(Y_test[0], test_predict[:,0])

## Lets compare testFinal actual and prediction values

pd.DataFrame(Y_testFinal[0], testFinal[:,0])

## Lets plot to visualize the actual and predicted values of testFinal
aa=[x for x in range(424)]
plt.figure(figsize=(15,4))
plt.plot(aa, Y_testFinal[0][:424], marker='.', label="actual")
plt.plot(aa, testFinal[:,0][:424], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
#plt.subplots_adjust(left=0.07)
plt.ylabel('pm25')
plt.xlabel('Time step')
plt.legend(fontsize=15)
plt.show();



#### As the model is build on Linear interpolation values which were imputed for 326 missing PM2.5 values. 

## The Accuracy is effected, also if we can get the other variables which impact PM2.5 values in the air, we can come up with better accuracy

## Slight Hyper-parameters optimization can be done reduce the error

model.save('C:\\Users\\abc\\Desktop\\AirPollution_Vinod\\LSTM_Model.h5')


