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
from utils import adf_test

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
coint_df = sample_df[['y2', 'y10']]
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
adf_test(coint_df['y2'], max_aug = 10)

# %%
adf_test(coint_df['y10'], max_aug = 10)

# %% [markdown]
# Results of ADF testing procedure for both series are shown above. In both cases null hypothesis is strongly rejected, and 6 augmentations is enough to get rid of residuals autocorrelation. We therefore conclude that both series are non-stationary.

# %%
coint_df['dy2'] = coint_df['y2'].diff()
coint_df['dy10'] = coint_df['y10'].diff()

# %%
coint_df.dropna(inplace = True)

# %%
adf_test(coint_df['dy2'], max_aug = 10)

# %%
adf_test(coint_df['dy10'], max_aug = 10)

# %% [markdown]
# However unit root tests have significant drawdown. Unit root specifically determines lack of mean reverting behavior of analyzed series. But constant mean is only 1 out of 3 conditions for weak stationarity. Financial time series often even after differencing exibits non-stationarity even if according to unit root test we would reject null hypothesis and assume stationarity. Take for example stock prices. Nominal changes in prices are proportional to the price, meaning that when stock price increases nominal changes increases as well implying non-constant variance. Therefore it may be better to first take logarithmic transformation of analyzed financial series. Then first differences of such series would be logarithmic returns, that usually can be treated approximately as percentage returns. 

# %%
fig, axs = plt.subplots(2, 2, figsize = (10, 4))

axs[0, 0].plot(np.log(coint_df['y2']).diff().dropna())
axs[0, 0].set_title('log diff - y2')

axs[0, 1].plot(coint_df['dy2'])
axs[0, 1].set_title('dy2')

axs[1, 0].plot(np.log(coint_df['y10']).diff().dropna())
axs[1, 0].set_title('log diff - y10')

axs[1, 1].plot(coint_df['dy10'])
axs[1, 1].set_title('dy10')

plt.tight_layout()

# %% [markdown]
# No significant difference between differenced series and differenced logarithm of the series is visible on the plots shown above. Therefore for simplicity we continue without logarithmic transformation.

# %% [markdown]
# ## ARIMA
