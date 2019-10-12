import numpy as np
import pandas as pd
import matplotlib.pylab as plt

#matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,6

#reading data
dataset=pd.read_excel('C:/Users/gurdsin/Desktop/ml/python/timeseries/weather dataset/weatherAUS.xlsx')

#pulling data for Sydney
dataset2 = dataset[dataset['Location']=='Sydney']
dataset2 = dataset2[['Date','Rainfall']]

#extracting month, day and year
dataset2['Date'] = pd.to_datetime(dataset2['Date'])
dataset2['Month'] = dataset2['Date'].dt.month
dataset2['Year'] = dataset2['Date'].dt.year
dataset2['Day'] = 1
dataset2['temp'] = '/'

#creating new date field
dataset2['Date_2'] = dataset2.apply(lambda row:str(row.Day)+row.temp+str(row.Month)+row.temp+str(row.Year), axis=1)
dataset3 = dataset2[['Date_2','Rainfall']]
dataset3['Date'] = pd.to_datetime(dataset3['Date_2'])
dataset3.drop(['Date_2'], axis=1, inplace=True)

#groupby the total rainfall monthly
dataset4 = dataset3.groupby('Date')['Rainfall'].sum()

rolmean = dataset4.rolling(window=12).mean()
rolstd = dataset4.rolling(window=12).std()

print(rolmean,rolstd)

#Plot Rolling Statistics
orig = plt.plot(dataset4,color='Blue',label='Original')
mean = plt.plot(rolmean,color='Red',label='Rolling Mean')
std = plt.plot(rolstd,color='black',label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean and Std Deviation')
plt.show(block=False)

# in the graph, the mean and the std devition is stable. So it is a stationary series
#converting series into dataframe
dataset4 = dataset4.to_frame()

#dataset4=dataset4.set_index(['Date'])
#Perform Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
print('Results of Dickey Fuller test:')
dftest=adfuller(dataset4['Rainfall'], autolag='AIC')
dfoutput=pd.Series(dftest[0:4], index={'Test Statistic','p-Value','#Lags Used','Number of Observations Used'})

for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key]=value

print(dfoutput)

#  p-value is less than 5%, which means we can reject the null hypothysis and conclude it is a stationary field

#Estimating Trend
# transformation: penalizing the higher value more than the smaller value
dataset4_logScale = np.log(dataset4)
plt.plot(dataset4_logScale)

# we can see forward trend in the data. So we used the same technique to estimate the trend and remove it from the series
movingAverage = dataset4_logScale.rolling(window=12).mean()
movingSTD = dataset4_logScale.rolling(window=12).std()
plt.plot(dataset4_logScale)
plt.plot(movingAverage,color='red')

#subtract the moving average from original value
datasetLogScaleMinusMovingAverage = dataset4_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#Removing NAN Values (i.e. first 12 months)
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):

#Determining Rolling Statistics
movingAverage = timeseries.rolling(window=12).mean()

#Plot Rolling Statistics
orig = plt.plot(timeseries,color='Blue',label='Original')
mean = plt.plot(movingAverage,color='Red',label='Rolling Mean')
std = plt.plot(movingSTD,color='black',label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean and Std Deviation')
plt.show(block=False)

#Perform Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
print('Results of Dickey Fuller test:')
dftest=adfuller(timeseries['Rainfall'], autolag='AIC')
dfoutput=pd.Series(dftest[0:4], index={'Test Statistic','p-Value','#Lags Used','Number of Observations Used'})
for key,value in dftest[4].items():
   dfoutput['Critical Value (%s)'%key]=value
print(dfoutput)

test_stationarity(datasetLogScaleMinusMovingAverage)

#Rolling values appear to mbe moving slightly but no specific tread. T-staistic is less than 5%, so we can say confidently that
#this series is stationary. This looks like a better series.

#Exponential Weighted Moving Meathod: more recent values given higher weights

exponentialDecayWeightedAverage = dataset4_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(dataset4_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')

# removing/subtracting the decays
datasetLogScaleMinusMovingExponentialDecayAverage = dataset4_logScale - exponentialDecayWeightedAverage

# Now less variation in std mean and deviation, t-statistic smaller than 1% critical value
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)

#Differencing: to reduce trend and seasonality. Here, we take difference in obs at perticular instant with that at
#previous instant. Thus improving stationarity

datasetLogDiffShifting = dataset4_logScale - dataset4_logScale.shift()
plt.plot(datasetLogDiffShifting)

# Removing NAs
datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)

#ACF and PACF plots - getting the q and p values from the graph
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

#Plot ACF
plt.subplot(121)
plt.plot(lag_acf)  
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
 
# p & q should be 2 as both the graphs cut 0 at 2
 
from statsmodels.tsa.arima_model import ARIMA

#AR Model
model = ARIMA(dataset4_logScale, order=(2,1,0))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues,color='Red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues - datasetLogDiffShifting["Rainfall"])**2))
print('Plotting AR Model')

#MA Model
model = ARIMA(dataset4_logScale, order=(0,1,2))
results_MA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues,color='Red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues - datasetLogDiffShifting["Rainfall"])**2))
print('Plotting MA Model')

#ARIMA Model
model = ARIMA(dataset4_logScale, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues,color='Red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - datasetLogDiffShifting["Rainfall"])**2))
print('Plotting MA Model')

# Residual Sum of Square is least in the ARIMA model, so we choose the ARIMA model
#  Now let's take it back to the original scale

#let's store our predictions in a series
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

# Convert difference to the log scale: add the difference consequently to the base number
#Get the cumulative Sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

#get the log values and convert it into series
predictions_ARIMA_log = pd.Series(dataset4_logScale['Rainfall'].ix[0], index=dataset4_logScale.index)

#add the cumulative sum to the values
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

# exp = e^n. Here e^log(n) = n

predictions_ARIMA= np.exp(predictions_ARIMA_log)
plt.plot(dataset4)
plt.plot(predictions_ARIMA)

# this predicts and plots the value. 12 years = 12*12 = 144. Rest are future values
results_ARIMA.plot_predict(1,264)

# forecasts value for next 120 periods
x=results_ARIMA.forecast(steps=120)