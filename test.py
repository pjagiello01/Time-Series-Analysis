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

# %%
adf_test(coint_df['y10'])

# %% [markdown]
# Now we can apply testing procedure to provide statistical evidence for cointegration. 

# %%

# %% [markdown]
# ## ARIMA

# %% [markdown]
#
