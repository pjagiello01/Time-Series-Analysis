# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Introduction

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from utils import *
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.api import OLS
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Turn off warnings
import warnings
warnings.filterwarnings('ignore')


# %% [markdown]
# ## Data
# <div style="width:800px; text-align:left; margin-left:0;">
# Given csv file with prices series for 10 different assets we need to find cointegrated pairs and choose one for the further analysis. We begin our analysis with visual inspection of the series.
#
# Recall that two series are cointegrated if both of them are $I(n)$, so integrated of order $n$, but some linear relationship of those series is $I(n-1)$, for $n \geq 1$. 
#
# With financial series we most often deal with series $I(1)$ and $I(0)$, so we expect to find 2 nonstationary series whose linear combination is stationary. Visually we expect that the difference between those series is fluctuating around some mean value with time. This is because:
# $$
# Y_t - \beta X_t = Z_t
# $$
# where $X_t$ and $Y_t$ are cointegrated series and $Z_t$ is stationary. As $Z_t$ is stationary, thus has constant mean over time, we should expect difference between $Y_t$ and $X_t$ fluctuating around some mean value.
# </div>

# %%
# Loading the data
df = pd.read_csv('data/prices.csv')

# %%
df.head()

# %%
df.tail()

# %%
df.info()

# %% [markdown]
# We can see above that the date column in our data is of object type representing string. It is recommended to work with dates as datetime objects set as index of the dateframe. So we change type of the data contained in date column to datetime and set the column as index.

# %%
df['date'] = pd.to_datetime(df['date'])
df.info()

# %%
df.set_index('date', inplace = True)
df.head()

# %%
# Split data
sample_df = df.iloc[:575]
sample_df.tail()

# %% [markdown]
# Now we can move on to the visual inspection. As we shown above we should expect difference between cointegrated series to fluctuate around some mean value. Starting from the top the behavior is clear for orange and ligth blue lines, so for y2 and y10 series.

# %%
# Visual inspection
plt.figure(figsize=(12, 6))
plt.plot(sample_df, label = sample_df.columns.values)
plt.grid(linestyle = '--', alpha = 0.4)
plt.legend(loc = 'upper left', fontsize = 'small')
plt.show()

# %%
coint_df = sample_df[['y3', 'y8']]
coint_df.head()

# %%
# Visual inspection
plt.figure(figsize=(12, 6))
plt.plot(coint_df, label = coint_df.columns.values)
plt.grid(linestyle = '--', alpha = 0.4)
plt.legend(loc = 'upper left', fontsize = 'small')
plt.show()

# %% [markdown]
# ## Stationatity
#
# Stationarity is an important concept in time series analysis. Weak stationarity assumes constant mean, variance and autocovariance over time. For our case statoinarity needs to be assesed for cointegration testing (check whether both series are $I(1)$) and to determine d order of ARIMA.
#
# #### Testing for stationarity
#
# Common way to check stationarity is visual analysis followed by statistical test like ADF test. We would expect stationary series to fluctuate over constant mean with irregular deviations of similar size. For both of our series we can clearly see some periods with upward trend and some with downward trend which implies no constant mean over time and non-stationarity of the series. 
#
# We can investigate this further with Augmented Dickey-Fuller test for unit root. ADF test statistic is derived assuming unit root existance (there exists root of the characteristic equation equal 1) implying non-stationarity. Test statistic does not follow any standard probability distribution and critical values were explicitly derived by Dickey and Fuller what is known as Dickey-Fuller table. For the statistical inference to be meaningful we need to make sure that the error term of model constructed for the test does not exhibit autocorrelation. For this purpose we simultaneously perform Breush-Godfrey tests for autocorrelation for some predetermined numbers of augmentations.
#
#

# %%
adf_test(coint_df['y3'], max_aug = 10)

# %%
adf_test(coint_df['y8'], max_aug = 10)

# %% [markdown]
# Results of ADF testing procedure for both series are shown above. In both cases null hypothesis is strongly rejected, and 6 augmentations is enough to get rid of residuals autocorrelation. We therefore conclude that both series are non-stationary.

# %%
coint_df['dy3'] = coint_df['y3'].diff()
coint_df['dy8'] = coint_df['y8'].diff()

# %%
coint_df.dropna(inplace = True)

# %%
adf_test(coint_df['dy3'], max_aug = 10)

# %%
adf_test(coint_df['dy8'], max_aug = 10)

# %% [markdown]
# However unit root tests have significant drawdown. Unit root specifically determines lack of mean reverting behavior of analyzed series. But constant mean is only 1 out of 3 conditions for weak stationarity. Financial time series often even after differencing exibits non-stationarity even if according to unit root test we would reject null hypothesis and assume stationarity. Take for example stock prices. Nominal changes in prices are proportional to the price, meaning that when stock price increases nominal changes increases as well implying non-constant variance. Therefore it may be better to first take logarithmic transformation of analyzed financial series. Then first differences of such series would be logarithmic returns, that usually can be treated approximately as percentage returns. 

# %%
fig, axs = plt.subplots(2, 2, figsize = (10, 4))

axs[0, 0].plot(np.log(coint_df['y3']).diff().dropna())
axs[0, 0].set_title('log diff - y3')

axs[0, 1].plot(coint_df['dy3'])
axs[0, 1].set_title('dy3')

axs[1, 0].plot(np.log(coint_df['y8']).diff().dropna())
axs[1, 0].set_title('log diff - y8')

axs[1, 1].plot(coint_df['dy8'])
axs[1, 1].set_title('dy8')

plt.tight_layout()

# %% [markdown]
# Logarithm of differenced series looks less stationary, therefore we continue with simple differences.

# %% [markdown]
# ## ARIMA
#
# We will apply Box-Jenkins procedure to find p,d,q values for ARIMA(p, d, q). This procedure utilizes ACF and PACF to determine correspondingly right MA order and right AR model. We already have order of integration as we have shown that series are both non-stationary but their differences are stationary implying integration of order 1 - $I(1)$

# %% [markdown]
# Let's start with procedure for y2 series. We shown that it is $I(1)$ so we plot ACF and PACF for differenced series.

# %%
fig, axes = plt.subplots(1, 2, figsize = (10, 4))

plot_acf(coint_df['dy3'], lags=20, ax=axes[0])
axes[0].set_title("ACF")

plot_pacf(coint_df['dy3'], lags=20, ax=axes[1]) 
axes[1].set_title("PACF")

plt.show()

# %% [markdown]
# We observe quite interesting behavior - ACF is decaying exponentially, which would be expected for simple AR(1) models. PACF on the other hand oscilates around 0, but also the greatest spike, which at the same time is significantly greater than other spikes is the first one (at lag 1). There is a possibility that the right model will be just AR(1) for differenced series, so ARIMA(1,1,0). Let's try this model

# %%
model = ARIMA(coint_df['y3'].values, order = (1,1,0))
arima_110 = model.fit()
print(arima_110.summary())

# %%
ljung_test = acorr_ljungbox(arima_110.resid, lags=[5, 10, 15, 20, 25], return_df=True)
print(ljung_test)

# %%
fig, axes = plt.subplots(1, 2, figsize = (10, 4))

plot_acf(arima_110.resid, lags=20, ax=axes[0]) 
axes[0].set_title("ACF")

plot_pacf(arima_110.resid, lags=20, ax=axes[1]) 
axes[1].set_title("PACF")

plt.show()

# %% [markdown]
# Model specification is correct. There is no autocorrelations in model residuals, and as usually we prefer simple models we could just take this one and move forward with the analysis as this model is basically the simplest one we can get. However, we can't really say that this model is certainly the best one without some comparison. It is therefore useful to study also different model specifications and compare them on information criteria basis (lower is better).
#
# ---

# %%
fig, axes = plt.subplots(1, 2, figsize = (10, 4))

plot_acf(coint_df['dy8'], lags=20, ax=axes[0])
axes[0].set_title("ACF")

plot_pacf(coint_df['dy8'], lags=20, ax=axes[1]) 
axes[1].set_title("PACF")

plt.show()

# %%
model = ARIMA(coint_df['y8'].values, order = (1,1,0))
arima_110_2 = model.fit()
print(arima_110_2.summary())

# %% [markdown]
# ---

# %% [markdown]
# ## Forecasting

# %%
sample_df.tail()

# %%
y3_forecast_df = get_forecast_df(n_steps=25, model=arima_110)
y3_forecast_df.index = df.tail(25).index

y8_forecast_df = get_forecast_df(n_steps=25, model=arima_110_2)
y8_forecast_df.index = df.tail(25).index

# %%
y3_df = df[['y3']].join(y3_forecast_df)
y8_df = df[['y8']].join(y8_forecast_df)

# %%
plot_forecast_with_ci('2025-03-03', y3_df, 'y3', 'forecast', 'lower ci', 'upper ci', 'y3 Forecast')

# %%
# Forecast evaluation
mape_y3 = mape(df['y3'][-25:], y3_forecast_df['forecast'])
amape_y3 = amape(df['y3'][-25:], y3_forecast_df['forecast'])
mse_y3 = mean_squared_error(df['y3'][-25:], y3_forecast_df['forecast'])
mae_y3 = mean_absolute_error(df['y3'][-25:], y3_forecast_df['forecast'])
rmse_y3 = np.sqrt(mse_y3)

# %%
print(f"""mape: {mape_y3}\n
amape: {amape_y3}\n
mse: {mse_y3}\n
rmse: {rmse_y3}\n
mae: {mae_y3}""")

# %%
plot_forecast_with_ci('2025-03-03', y8_df, 'y8', 'forecast', 'lower ci', 'upper ci', 'y8 Forecast')

# %%
# Forecast evaluation
mape_y8 = mape(df['y8'][-25:], y8_forecast_df['forecast'])
amape_y8 = amape(df['y8'][-25:], y8_forecast_df['forecast'])
mse_y8 = mean_squared_error(df['y8'][-25:], y8_forecast_df['forecast'])
mae_y8 = mean_absolute_error(df['y8'][-25:], y8_forecast_df['forecast'])
rmse_y8 = np.sqrt(mse_y8)

print(f"""mape: {mape_y8}\n
amape: {amape_y8}\n
mse: {mse_y8}\n
rmse: {rmse_y8}\n
mae: {mae_y8}""")

# %%
