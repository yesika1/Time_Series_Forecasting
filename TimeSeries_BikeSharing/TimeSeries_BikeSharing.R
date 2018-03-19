# ---------------------------------------------------------------------------------------- #
#
# TimeSeries_BikeSharing.R 
# Author: Yesika Contreras
#  
# Exponential smoothing & ARIMA time series models to predict the demand of bike rentals.
#
# Bike Sharing Dataset
# Dataset on the daily number of bicycles checkouts from a bike sharing service, 
# being part of the UCI Machine Learning Repository
# 
# Process following Ruslana Dalinina, with subtles additions as:
# - Test for additive or multiplicative seasonality
# - Splitting in train and test data before the generation of the models.
# - Adding an Exponential smoothing approach for forecasting
# Ruslana Dalinina approach available: https://www.datascience.com/blog/introduction-to-forecasting-with-arima-in-r-learn-data-science-tutorials
#
# R scripts generated 
# Sys.Date() # "2018-03-16"
#
# ---------------------------------------------------------------------------------------- #

#============================================================
# Libraries
#============================================================

library(ggplot2)
library(forecast)
library(tseries)
library(TStools)

#if (!require("devtools")){install.packages("devtools")}
#devtools::install_github("trnnick/TStools")

#============================================================
# Importing data
#============================================================

#setwd('path')
daily_df <- read.csv('day.csv', header=TRUE, stringsAsFactors=FALSE)
head(daily_df)
str(daily_df) #731 obs. of  19 variables
summary(daily_df)


#============================================================
# Cleaning data
#============================================================

# Transforming dteday:date to Date type
daily_df$dteday = as.Date(daily_df$dteday)


# Looking for any outliers, volatility, or irregularities. 
sum(is.na(daily_df)) #0

# ploting rental bikes per day (Demand)
#-------------------------------

count_plot <- ggplot(daily_df, aes(dteday, cnt)) +
          geom_line() +
          scale_x_date('month') +
          ggtitle('Daily Bike Checkouts') +
          ylab('Rental Bicycle count ')
          
# cnt: count of total rental bikes including both casual and registered
# Possible Outliers: In some cases, the number of bicycles checked out dropped below 100 
# on day and rose to over 4,000 the next day.


#============================================================
# Transforming data to time series (ts)
#============================================================

# Creating a time series object with ts() and passing tsclean()
# tsclean() identifies and replaces outliers using series smoothing and decomposition.
# tsclean() also inputes missing values in the series if there are any

count_ts <- ts(daily_df$cnt)
# ts(df$col, frequency=dataFreq, start=startEntry) #create a time series
# plot (tsclean(count_ts), main= 'Cleaned Rental Bicycle Count' )

daily_df$clean_cnt <- tsclean(count_ts)


clean_count_plot <- ggplot(daily_df, aes(dteday, clean_cnt)) +
                  geom_line(color= 'cornflowerblue') + 
                  ggtitle('Daily Bike Checkouts II') +
                  ylab('Cleaned Rental Bicycle Count')

# After cleaning: 
# Even after removing outliers, the daily data is still pretty volatile. Visually, 
# we could a draw a line through the series tracing its bigger troughs and peaks while 
# smoothing out noisy fluctuations (moving average). 
# The wider the window of the moving average, the smoother original series becomes.


# Smoothing the series with weekly or monthly moving average (ma)
# -------------------------------

# Taking weekly or monthly moving average, 
# smoothing the series into something more stable and therefore predictable 
# # High-order (level) aggregation: A simple option is to convert a TS to higher-order, like quarters instead of months, years instead of quarters. 

# Centered moving averages: 
# Main idea is to average values of TS within k periods of t, 
# with m is the moving average order. m=2k+1  
# so, with MA=m we have k observations at the beginning and end, using the observation in between for averaging. 
# we usually use odd numbers because even numbers is an average of two asymmetric moving averages. 


daily_df$cnt_ma7 <- ma(daily_df$clean_cnt, order=7) # order =daily using the clean count with no outliers # generates Na's=6, m7 is centered in the 4 data, which is the first value to appear.
daily_df$cnt_ma30 <- ma(daily_df$clean_cnt, order=30)

ma_plot <- ggplot() +
  geom_line(data = daily_df, aes(x = dteday, y = clean_cnt, color = "Daily Counts")) +
  geom_line(data = daily_df, aes(x = dteday, y = cnt_ma7,   color = "Weekly Moving Average"))  +
  geom_line(data = daily_df, aes(x = dteday, y = cnt_ma30, color = "Monthly Moving Average"))  +
  ylab('Bicycle Count')

#Selecting time framework for modeling:
# The higher the window parameter, the more we capture the global trend, but miss on local trends 
# We will model the smoothed series of weekly moving average (blue line). ts = cnt_ma7



#============================================================
# Decomposing Data
#============================================================

# Decomposing data is the process of extracting the components of seasonality, trend, and cycle. 
# Deconstructing the series can help to understand the behavior 
# before building a forecasting model.

# Seasonality: fluctuations in the data related to calendar cycles. 
# For example, more people might be riding bikes in the summer, and less during colder months.
# Usually, seasonality is fixed value; for instance, quarter(4) or month of the year.

# Trend component is the overall pattern of the series: Is the number of bikes rented increasing or decreasing over time.
  
# Cycle component consists of decreasing or increasing patterns that are not seasonal. 
# Usually, trend and cycle components are grouped together & estimated using moving averages.

# Residual or error: part of the series that can't be explained/attributed to the components.

# ARIMA models can be fitted to both seasonal and non-seasonal data. 
# Seasonal ARIMA requires a more complicated specification of the model structure.
# We will explore how to de-seasonalize the series and use a non-seasonal ARIMA model.


# Decomposing time series using stl().
# -------------------------------

# It calculates the seasonal component of the series using smoothing, and
# adjusts the original series by subtracting seasonality 
# stl() by default assumes additive model structure. 
# Use allow.multiplicative.trend=TRUE to incorporate the multiplicative model.

# Using the smoothed ts (cnt_ma7) we generate a new ts with 30 days frequency before decomposing
count_ma7 <- ts(na.omit(daily_df$cnt_ma7), frequency=30) #Set frequency to 12 for monthly, 4 for quaterly data, & 30 for daily. 
decomp = stl(count_ma7, s.window="periodic") # s.window is the seasonal window=speed of seasonal changes, "periodic" is for constant seasonal effects.
plot(decomp, main='Decomposed time series')


# Calculate de-seasonal component using seasadj()
# ------------------------------
deseasonal_cnt <- seasadj(decomp) # de-seasonal time series. We will modeling with this set.


# Testing Stationarity with the augmented Dickey-Fuller test
# -------------------------------

# Fitting an ARIMA model requires the series to be stationary. 
# A series is stationary when its mean, variance, and autocovariance are time invariant.
# Modeling stable series with consistent properties involves less uncertainty.

# The augmented Dickey-Fuller (ADF) test for stationarity. 
# The null hypothesis assumes that the series is non-stationary (there is a presence of a trend component). 


adf.test(count_ma7, alternative = "stationary") # on ts with freq 30 before decomp
# Dickey-Fuller = -0.2557, Lag order = 8, p-value = 0.99
# alternative hypothesis: stationary
# Since  p-value = 0.99 > 0.05 we cannot reject the hypotesis that there is not difference (ho).
# Then, A formal ADF test does not reject the null hypothesis of non-stationarity.

# The bicycle data is non-stationary. The average number of bike checkouts changes over time. 
# We can observe the trend in the decomposed plot

# Trend means a consistent slope direction
# (1) Additive trend: 
# Y(t) = Trend(t) + Season(t) + Error(t)
# (2) Multiplicative trend: 
# Y(t) = Trend(t) * Season(t) * Error(t)
# (3) Stationarity: no observed trend, only errors. 


# Testing multiplicative or Additive seasonality
# ---------------------------------

test_multiplicative <- TStools::mseastest(count_ma7)
test_multiplicative$is.multiplicative # FALSE
# Then the ts is additive



#============================================================
# Autocorrelations Plots and Choosing Model Order
#============================================================

# Usually, non-stationary series can be corrected by a simple transformation such as differencing.
# Differencing the series can help in removing its trend or cycles.
# order of differencing: The number of differences performed is represented by the d component of ARIMA.

# Autocorrelation plots (ACF) visual tool in determining whether a series is stationary. 
# If the series is correlated with its lags then, generally, there are some trend or seasonal components.
# ACF plots display correlation between a series and its lags. 
# ACF plots can help in determining the order of the M A (q) model. 
# Partial autocorrelation plots (PACF), display correlation between a variable and its lags 
# that is not explained by previous lags. PACF plots are useful when determining the order of the AR(p) model.

par(mfrow=c(1,2))
Acf(count_ma7, main='Autocorrelation plot (ACF)')
Pacf(count_ma7, main='Partial autocorrelation plot (PACF)')
dev.off()

# ACF plot shows that the ts has significant autocorrelations with many lags.
# PACF plot only shows a spike at lags 1 and 7.
# Then the ACF autocorrelations could be due to carry-over correlation from the first or early lags


# Differencing until getting stationarity
# -------------------------------

# Start differencing the deseasonal ts with the order of d = 1 and re-evaluate if further differencing is needed.
count_differenced1 = diff(deseasonal_cnt, differences = 1)

#plotting differenced ts
plot(count_differenced1, main = 'Differencing the De-seasonalized Time Series ')

# Evaluating stationarity
adf.test(count_differenced1, alternative = "stationary")
# Augmented Dickey-Fuller Test, data:  count_differenced1
# Dickey-Fuller = -9.9255, Lag order = 8, p-value = 0.01
# alternative hypothesis: stationary

# Since  p-value = 0.01 < 0.05 we can reject the hypotesis that there is not difference (ho).
# Then, A formal ADF test rejects the null hypothesis of non-stationarity on differenced data.
# Now the data transformed looks stationary, there is not visible strong trend. 
# Thus, Differencing of order 1 terms is sufficient and should be included in the model.

# Choosing p & q for the model:
# -------------------------------

# spikes at particular lags of the differenced series can help to choose of p or q for our model.
par(mfrow=c(1,2))
Acf(count_differenced1, main='ACF for Differenced Time Series')
Pacf(count_differenced1, main='PACF for Differenced Time Series')
dev.off()

# From ACF: There are significant auto correlations at lag 1, 2 and 7,8. 
# From PACF: Partial correlation plots show a significant spike at lag 1 and 7. 
# Conclusion: test models with AR or MA components of order 1, 2, or 7.
# A spike at lag 7 might suggest that there is a seasonal pattern present, perhaps as day of the week. 



#============================================================
# Fitting an ARIMA model
#============================================================


# arima(): Manually specify the order parameters of the model
# auto.arima(): automatically generate a set of optimal (p, d, q) that optimizes model fit criteria.
# auto.arima() also allows the user to specify maximum order for (p, d, q), which is set to 5 by default.


# Splitting data in training and testing using the window() function
# ----------------

# We are going to leave as testing set the same window that we want to predit (~30 days)

test_Arima <- window(ts(deseasonal_cnt), start=700)
train_Arima <- window(ts(deseasonal_cnt), start=1, end=699)
#plot(train_Arima)

model_Arima_train <- auto.arima(train_Arima, seasonal=TRUE) # there is presence of seasonal component in original data
# Series: train_ts 
# ARIMA(2,1,0) 
# Coefficients:
#     ar1     ar2
#     0.2557  0.0870
#s.e. 0.0377  0.0378
#sigma^2 estimated as 25631:  log likelihood=-4532.36
#AIC=9070.72   AICc=9070.76   BIC=9084.37
 

#  Arima (p, d, q) components= p: AR order, d: degree of differencing, q: MA order
# A good way to think about it is (AR, I, MA)  
# Y = (Auto-Regressive Parameters) + (Moving Average Parameters), the 'I' part of the model (the Integrative part) 
# Arima(2,1,0):  the model uses an autoregressive term of second lag,
# incorporates differencing of degree 1, and a moving average model of order 0.
# AR(1) coefficient p tells us that the next value in the series is taken as a dampened previous value by a factor of x 

# (AIC): Akaike information criteria 
# (BIC): Baysian information criteria 



#============================================================
# Model Evaluation
#============================================================

# Examining ACF and PACF plots for model residuals. 
# If model order parameters and structure are correctly specified, we would expect no significant autocorrelations present. 

tsdisplay(residuals(model_Arima_train), lag.max=31, main='ARIMA (2,1,0) Model Residuals')
# There is a clear pattern present in ACF/PACF and model residuals plots repeating at lag 7. 
# This suggests that our model may be better off with a different specification, such as p = 7 or q = 7.



model_Arima2 <- arima(train_Arima, order=c(2,1,7))
# Call: arima(x = train_ts, order = c(2, 1, 7))
# Coefficients:
#     ar1     ar2     ma1     ma2     ma3     ma4     ma5     ma6      ma7
#     0.2564  0.0188  0.1298  0.1302  0.1017  0.1028  0.1083  0.1293  -0.8545
#s.e. 0.0459  0.0435  0.0282  0.0271  0.0271  0.0277  0.0256  0.0260   0.0307
#sigma^2 estimated as 13942:  log likelihood = -4330.96,  aic = 8681.93

tsdisplay(residuals(model_Arima2), lag.max=31, main='ARIMA (2,1,7) Model Residuals')
# The model uses an autoregressive term of second lag,
# incorporates differencing of degree 1, and a moving average model of order 7.
# Result: There are no significant autocorrelations present in the residuals. The model was correctly specified.

accuracy(model_Arima2)
#               ME     RMSE      MAE        MPE     MAPE      MASE      ACF1
# Training set 4.768835 117.9898 87.72782 0.125315 2.191441 0.7040459 -0.0004262534



#============================================================
# Forecasting
#============================================================


# We can specify forecast horizon h periods ahead for predictions to be made.
# We are going to predict the values for the next month, h= 25

forecast_test <- forecast(model_Arima2,h=25)
plot(forecast_test, main= ' Forecasting using Arima Model')
lines(ts(deseasonal_cnt))


# forecast estimates are provided with confidence bounds: 80% confidence limits shaded in darker blue, 
# and 95% in lighter blue. 
# Longer term forecasts will usually have more uncertainty,which is reflected in the wider confidence bounds when time progress. 
# The pattern in confidence bounds may signal the need for a more stable model (lower expected error associated with the estimations). 



# Future work: 
# -------------
# Generating an exponential smoothing model, would help make the model more accurate using a weighted combinations of seasonality, trend, and historical values to make predictions. 
# On the other hand, daily bicycle demand probably dependend on other factors (weather, holidays, time of the day) that could be addressed with an ARMAX or dynamic regression models.



#============================================================
# Exponential Smoothing Approach - Holt-Winters
#============================================================

# to triple exponential (Holt-Winters) that addresses level, trend and season. 
# For all three components, select model types with three letters: 
# A = additive
# M = multiplicative
# N = none
# Z = automatic selection 


test_Hw <- ts(count_ma7[c(700:725)],frequency = 30)
train_Hw <- ts(count_ma7[-c(700:725)],frequency = 30)

# using ets: Error, Trend, Seasonality
model_HoltWinters <- HoltWinters(train_Hw,seasonal="additive")

# Holt-Winters exponential smoothing with trend and additive seasonal component.
#Call:  HoltWinters(x = train_Hw, seasonal = "additive")
# Smoothing parameters:
#  alpha: 0.9431843
# beta : 0.002067271
# gamma: 1

tsdisplay(residuals(model_HoltWinters), lag.max=31, main='Holt-winters Model Residuals')

qqnorm(model_HoltWinters$residuals,  main="Q-Q plot. Residuals model Holt-Winters")
qqline(model_HoltWinters$residuals, col= "orange")


accuracy(model_HoltWinters) 
#               ME     RMSE      MAE        MPE     MAPE      MASE      ACF1
# Training set 3.103773 164.8497 118.5047 0.1005193 2.937568 0.9541444 0.04234134


forecast_HoltWinters <- forecast(model_HoltWinters2,h=25)
plot(forecast_HoltWinters, main= ' Forecasting using HoltWinters')
lines(ts(count_ma7))


#============================================================
# Comparing models
#============================================================

#Plotting Forecasting: 
par(mfrow=c(2,1))
plot(forecast_test, main= ' Forecasting using Arima Model')
lines(ts(deseasonal_cnt))

plot(forecast_HoltWinters, main= ' Forecasting using HoltWinters Model')
lines((count_ma7))

dev.off()


