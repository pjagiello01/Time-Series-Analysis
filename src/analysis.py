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
# ## TASK
#
# The aim of the project is to compare accuracy of forecasts of prices of two cointegrated financial
# instruments with VECM model and two independent univariate ARIMA models.
# - Find one cointegrated pair out of ten provided time series. There is more than one cointegrated
# pair but you are supposed to find just one of them. If you found more than one pair, you can
# choose any of them for further analysis.
# - Build a bivariate VECM model for the identified pair and produce forecasts of prices of two
# instruments for the out-of-sample period.
# - Find separately for two instruments the most attractive univariate ARIMA models and produce
# forecasts for the same out-of-sample period.
# - Compare accuracy of forecasts of the prices using the ex-post forecast error measures.
# - Prepare a short report on it.

# %%
# Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from utils import *
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR, VECM
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.api import OLS
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm

# Turn off warnings
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Data
#
# Given csv file with prices series for 10 different assets we need to find cointegrated pairs and choose one for the further analysis. We begin our analysis with visual inspection of the series.
#
# Recall that two series are cointegrated if both of them are $I(n)$, so integrated of order $n$, but some linear relationship of those series is $I(n-1)$, for $n \geq 1$. 
#
# With financial series we most often deal with series $I(1)$ and $I(0)$, so we expect to find 2 nonstationary series whose linear combination is stationary. Visually we expect that the difference between those series is fluctuating around some mean value with time. This is because:
# $$
# Y_t - \beta X_t = Z_t
# $$
# where $X_t$ and $Y_t$ are cointegrated series and $Z_t$ is stationary. As $Z_t$ is stationary, thus has constant mean over time, we should expect difference between $Y_t$ and $X_t$ fluctuating around some mean value.

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
test_df = df.iloc[575:]
sample_df.tail()


# %% [markdown]
# Now we can proceed with visual inspection and cointegration testing. 
#
# ---

# %% [markdown]
# ## Testing for cointegration

# %% [markdown]
# ### Visual analysis
#
# As we shown above we should expect difference between cointegrated series to fluctuate around some mean value. Starting from the top the behavior is clear for orange and ligth blue lines, so for y2 and y10 series.

# %%
plt.figure(figsize=(12, 6))
plt.plot(sample_df, label = sample_df.columns.values)
plt.grid(linestyle = '--', alpha = 0.4)
plt.legend(loc = 'upper left', fontsize = 'small')
plt.show()

# %% [markdown]
# On the chart above multiple potentially cointegrated pairs are visible. We choose y3 and y8.

# %%
coint_df = sample_df[['y3', 'y8']]
coint_test_df = test_df[['y3', 'y8']]
coint_df.head()

# %%
plt.figure(figsize=(12, 6))
plt.plot(coint_df, label = coint_df.columns.values)
plt.grid(linestyle = '--', alpha = 0.4)
plt.legend(loc = 'upper left', fontsize = 'small')
plt.show()

# %% [markdown]
# Clearly we can observe long term relationship between those two time series, where for most of the time Y8 is above Y3, with that behaviour changing around 2024-10. In further steps we will examine this relationship more formally by running tests for cointegration.

# %% [markdown]
# ### Engle-Granger Two-Step approach

# %% [markdown]
# As we already mentioned, series are cointegrated when both are $I(n)$, but their linear combination is $I(n-1)$. Most often we deal with series $I(1)$, therefore one way to check if series are corerlated is to test whether both series are $I(1)$ and if there is some linear combination of them which is $I(0)$.
#
# This procedure consists of two steps as its name suggests:
# 1. Testing both series for stationarity. ADF test can be used.
# 2. Fitting linear regression model and testing stationarity of residuals.
#

# %% [markdown]
# #### Stationatity
#
# Stationarity is an important concept in time series analysis. Weak stationarity assumes constant mean, variance and autocovariance over time. For our case statoinarity needs to be assesed for cointegration testing (check whether both series are $I(1)$) and to determine d order of ARIMA.
#
# ##### Testing for stationarity
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
# Results of ADF testing procedure for both series are shown above. In both cases null hypothesis is strongly rejected, and 3 augmentations is enough to get rid of residuals autocorrelation. We therefore conclude that both series are non-stationary.

# %%
# Introduce first differences

coint_df['dy3'] = coint_df['y3'].diff()
coint_df['dy8'] = coint_df['y8'].diff()

# %%
coint_df.dropna(inplace = True)

# %%
adf_test(coint_df['dy3'], max_aug = 10)

# %%
adf_test(coint_df['dy8'], max_aug = 10)

# %% [markdown]
# Results above show that null hypothesis stating unit root existance is strongly rejected. Therefore we conclude that both series y3 and y8 are $I(1)$
#
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
# #### Linear regression

# %%
# OLS estimation with constant, 
X = sm.add_constant(coint_df['y8'])
model_ols = OLS(coint_df['y3'], X).fit()

# %%
model_ols.summary()

# %%
# Residuals plot
plt.figure(figsize = (12, 6))
plt.plot(model_ols.resid, label = 'Residuals')
plt.grid(linestyle = '--', alpha = 0.3)
plt.show()

# %%
# Residuals ADF test
adf_test(model_ols.resid, max_aug = 10)

# %% [markdown]
# Null hypothesis of ADF test is strongly rejected, thus we conclude that series y3 and y8 are cointegrated. From OLS estimates we get normalized cointegration vector. It is $[1, -44.16, -0.625]$, where $1$ is y3 coefficient, $44.16$ is constant and $-0.625$ is y8 coefficient:
# $$
# y_{3_t} - 44.16 - 0.625 y_{8_t} = \epsilon_t \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)
# $$

# %% [markdown]
# ### Johansen test
#
#  $ΔYt = ΠYt−1 + Γ1ΔYt−1 + Γ2ΔYt−2 + . . . + Γp−1ΔYt−(p−1) + εt$

# %% [markdown]
# We would like to run Johansen Cointegration tests. But because lag length can affect this test, we should first decide how many lags will be taken into consideration. Because we are dealing with financial asset prices first what comes to mind is 5 lags. But to make more informed decision we can employ information criteria.

# %%
model_var = VAR(coint_df[['y3', 'y8']], freq='D')
results = model_var.select_order(maxlags=10)
print("\nLag selection results:")
print(results.summary())

# %% [markdown]
# We are looking at lowest values for each criteria. We can see that two of the information criteria are pointing to 3 lags and two to 4. Let's use method for picking up the most optimal number of lags.

# %%
selct_var_order(coint_df[['y3', 'y8']], max_lags=10)

# %% [markdown]
# Given that we are dealing with financial instruments data we will use 3 number of lags in our test. Also AIC and FPE usually favor bigger number of lags, BIC is usually on the lower side, but HQIC should be something in between.
#

# %% [markdown]
# #### Performing Johansen test
#
# We are performing this test to verify if cointegrated vector exists for those two time series.

# %%
# Perform Johansen test
# K=4 in levels VAR -> k_ar_diff = K-1 = 3 lags in VECM differences
# ecdet = "const" -> det_order = 0 (constant in CE)

johansen_result = coint_johansen(sample_df[['y3', 'y8']], det_order=0, k_ar_diff=3)

print("Johansen Test Results:")
print("Eigenvalues:")
print(johansen_result.eig)
print("\nTrace Statistic:")
print(johansen_result.lr1)
print("\nCritical Values (90%, 95%, 99%) for Trace Statistic:")
print(johansen_result.cvt)
print("\nMaximum Eigenvalue Statistic:")
print(johansen_result.lr2)
print("\nCritical Values (90%, 95%, 99%) for Max Eigenvalue Statistic:")
print(johansen_result.cvm)

# %%
print("--- Interpretation (Trace Test) ---")
hypotheses_trace = ['r <= 0', 'r <= 1']
for i in range(len(hypotheses_trace)):
    print(f"H0: {hypotheses_trace[i]}")
    print(f"  Trace Statistic: {johansen_result.lr1[i]:.3f}")
    print(f"  Critical Value (95%): {johansen_result.cvt[i, 2]:.3f}")
    if johansen_result.lr1[i] > johansen_result.cvt[i, 2]:
        print("  Result: Reject H0 at 5% significance level.")
    else:
        print("  Result: Cannot reject H0 at 5% significance level.")

print("\n--- Interpretation (Max Eigenvalue Test) ---")
hypotheses_maxeig = ['r = 0', 'r = 1']  # H0: rank is r vs H1: rank is r+1
for i in range(len(hypotheses_maxeig)):
    print(f"H0: {hypotheses_maxeig[i]}")
    print(f"  Max Eigenvalue Statistic: {johansen_result.lr2[i]:.3f}")
    print(f"  Critical Value (95%): {johansen_result.cvm[i, 2]:.3f}")
    if johansen_result.lr2[i] > johansen_result.cvm[i, 2]:
        print("  Result: Reject H0 at 5% significance level.")
    else:
        print("  Result: Cannot reject H0 at 5% significance level.")

# %% [markdown]
# In both cases for Trace and Max Eigenvalue tests we can see that H0 indicating that there is **no cointegrating vector can be rejected**. However, we fail to reject H0 that there is **one cointegrating vector** thus implying that we have exactly one cointegrating vector and series is cointegrated.
#
# ---

# %% [markdown]
# ## VECM model
#
# Following variables will be used in our VECM model
# * Previously set number of lags = 4 so lag order in differences will be 3.
# * Cointegration as already establish is of rank one
# * Daily frequency = daily
# * why ci exactly?

# %%
# Estimate VECM
vecm_model = VECM(coint_df[['y3', 'y8']], k_ar_diff=3, coint_rank=1, deterministic='ci', freq='D')
vecm_results = vecm_model.fit()

# %% [markdown]
# Let's see the results:

# %%
print(vecm_results.summary())

# %% [markdown]
# The VECM is given by:
#
# $$\Delta\boldsymbol{y}_t = \Pi\boldsymbol{y}_{t-1} + \Gamma_1\Delta\boldsymbol{y}_{t-1} + ... + \Gamma_p\Delta\boldsymbol{y}_{t-p} + \boldsymbol{c}_d + \boldsymbol{\varepsilon}_t$$
#
# where long run coefficient matrix is described as:
# $$
# \boldsymbol{\Pi}=\alpha\beta'=\left[
# 	\begin{array}{ccc}
# 	\alpha_{11} \\
# 	\alpha_{21} \\
# 	\end{array}
# \right]
# \left[
# 	\begin{array}{cccc}
# \beta_{11} \quad \beta_{21} \\
#  \end{array}
# \right]
# $$
#

# %% [markdown]
# let's see what values it takes in our case:

# %%
print("\nLong-run coefficient matrix:")
print(f"{vecm_results.alpha} {vecm_results.beta[:, 0] / vecm_results.beta[0, 0]}")

# %% [markdown]
# **Interpretations:**
#
# Alphas ( 'adjustment parameters' )
# * $\alpha_{11}$ - we see it being positive, and in this case we would expect negative sign, but coefficient is not statistically significant.
# * $\alpha_{12}$ - is positive 2.5163 and statistically significant, meaning that correction mechanism works in expected direction. Variable should return to the long-term equilibrium - when it's above it should adjust downwards and below it adjust upwards.
#
#  Betas ( coingegrating vectors )
# * $\beta_{11}$ - normalization applied thanks to decomposition of Π not being unique.
# * $\beta_{12}$ - negative sign tells us that y$_{1}$ it is positively related to y$_{2}$. In perfect equilibrium we would have y$_{1}$ = 0.625 * y$_{2}$.

# %% [markdown]
# Lag terms:
# $$\Gamma_1\Delta\boldsymbol{y}_{t-1} + ... + \Gamma_p\Delta\boldsymbol{y}_{t-p} + \boldsymbol{c}_d + \boldsymbol{\varepsilon}_t$$
#
# this represents short-run dynamics outside cointegration relation. None of the coefficients is statistically significant thus we are ommiting interpretation.
#
# ---

# %% [markdown]
# **We are reparametrizing VECM to VAR for further examination of the relations**

# %%
# Estimate VAR
var_model = VAR(coint_df[['y3', 'y8']], freq='D')
var_results = var_model.fit(4)

# %% [markdown]
# #### Impulse Response Functions
# Let's examine what effect have shocks applied to variables. We are applying 160 periods as variable to illustrate correction mechanism more clearly.

# %%
# Calculate and plot impulse response functions
irf = var_results.irf(160)  # 160 periods ahead
# and hence interpretations may change depending on variable ordering.
irf.plot(orth=True)

# %% [markdown]
# **Interpretation**
#
# * y3->y3 : initial increase up to 25 days, and afterward slow decrease toward equilibrium
# * y3->y8 : shock has bigger initial impact on y8, but as in previous case impacted variable is slowly converging toward 0
# * y8->y3 : no first period impact of the y8 on y3. But we can observe negative impact on y3 and which converges back to 0
# * y8->y8 : quick small jump at the beginning but as in previous case we are observing negative impact on first periods and convergence starting prior 25 period.

# %% [markdown]
# #### Forecast Error Variance Decomposition
#
# FEVD can help us to see what is the proportion of variance in error terms between each variable due to their own shock or shocks of other variable.

# %%
# Calculate and plot forecast error variance decomposition
fevd = var_results.fevd(25)  # 36 periods ahead
fevd.summary()

# Plot FEVD
var_results.fevd(25).plot()

# %% [markdown]
# We can see that variance of forecast error terms are explained mainly by changes in y3 for both variables. That confirms our main previous observations that shocks in y3 have impact on y8 but not other way around.

# %% [markdown]
# #### Autocorrelation in residuals

# %%
residuals = var_results.resid

residuals = pd.DataFrame(residuals, columns=['y3', 'y8'])

# Test for serial correlation (Ljung-Box test on residuals)
# Apply to each residual series separately
print("\nLjung-Box Test for Serial Correlation in Residuals (lag=10):")
ljung_box_y3 = acorr_ljungbox(residuals['y3'], lags=[10], return_df=True)
ljung_box_y8 = acorr_ljungbox(residuals['y8'], lags=[10], return_df=True)
print("y3 Residuals:\n", ljung_box_y3)
print("\ny8 Residuals:\n", ljung_box_y8)

# Check p-values
alpha_serial = 0.05
print(f"\nConclusion at alpha={alpha_serial}:")
if ljung_box_y3['lb_pvalue'].iloc[0] < alpha_serial:
    print(" - Reject H0 (no serial correlation) for y3 residuals.")
else:
    print(" - Cannot reject H0 (no serial correlation) for y3 residuals.")
if ljung_box_y8['lb_pvalue'].iloc[0] < alpha_serial:
    print(" - Reject H0 (no serial correlation) for y8 residuals.")
else:
    print(" - Cannot reject H0 (no serial correlation) for y8 residuals.")

# %% [markdown]
# Test indicate that there is no significant autocorrelation in residuals of chosen VAR model with 10 lags.
# Let's examine how relation between residuals looks on diagram:

# %%
# Plot ACF and PACF of residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_acf(residuals['y3'], ax=axes[0, 0], title='ACF - y3 Residuals', lags=20)
plot_pacf(residuals['y3'], ax=axes[1, 0], title='PACF - y3 Residuals', lags=20,
          method='ywm')  # 'ols' method might fail if near non-stationarity
plot_acf(residuals['y8'], ax=axes[0, 1], title='ACF - y8 Residuals', lags=20)
plot_pacf(residuals['y8'], ax=axes[1, 1], title='PACF - y8 Residuals', lags=20, method='ywm')
plt.tight_layout()
plt.show()

# %% [markdown]
# ACF and PACF for both variables shows that there might be some autocorrelation in 4th 6th and 17th lags.
#
# Let's see how those would look like if we would add upt to 6th lag to our model.

# %%
var_model_6 = VAR(coint_df[['y3', 'y8']], freq='D')
var_results_6 = var_model_6.fit(7)
resid_6 = var_results_6.resid

resid_6 = pd.DataFrame(resid_6, columns=['y3', 'y8'])

# Plot ACF and PACF of residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_acf(resid_6['y3'], ax=axes[0, 0], title='ACF - y3 Residuals', lags=20)
plot_pacf(resid_6['y3'], ax=axes[1, 0], title='PACF - y3 Residuals', lags=20,
          method='ywm')  # 'ols' method might fail if near non-stationarity
plot_acf(resid_6['y8'], ax=axes[0, 1], title='ACF - y8 Residuals', lags=20)
plot_pacf(resid_6['y8'], ax=axes[1, 1], title='PACF - y8 Residuals', lags=20, method='ywm')
plt.tight_layout()
plt.show()

# %% [markdown]
# Incorporating more lags we removed autocorrelation between residuals. But given that the difference doesn't look that significant we won't respecify our previous model.

# %% [markdown]
# **Proceeding to checking normality of residuals:**

# %%
# Plot histograms of residuals with normal density overlay
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(residuals['y3'], kde=False, stat='density', ax=axes[0], color='pink', edgecolor='black')
axes[0].set_title('Density of y3 Residuals')
# Overlay normal distribution
mu_y3, std_y3 = residuals['y3'].mean(), residuals['y3'].std()
xmin_y3, xmax_y3 = axes[0].get_xlim()
x_y3 = np.linspace(xmin_y3, xmax_y3, 100)
p_y3 = norm.pdf(x_y3, mu_y3, std_y3)
axes[0].plot(x_y3, p_y3, 'k', linewidth=2, label='Normal Fit')
axes[0].legend()

sns.histplot(residuals['y8'], kde=False, stat='density', ax=axes[1], color='pink', edgecolor='black')
axes[1].set_title('Density of y8 Residuals')
# Overlay normal distribution
mu_y8, std_y8 = residuals['y8'].mean(), residuals['y8'].std()
xmin_y8, xmax_y8 = axes[1].get_xlim()
x_y8 = np.linspace(xmin_y8, xmax_y8, 100)
p_y8 = norm.pdf(x_y8, mu_y8, std_y8)
axes[1].plot(x_y8, p_y8, 'k', linewidth=2, label='Normal Fit')
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# Distribution is only a little bit denser in the middle, but we don't see huge tails or other characteristics of non-normal distribution. We will examine normality of distributions by running Jarque-Bera tests.

# %%
# Test for normality (Jarque-Bera test)
jb_y3_stat, jb_y3_pval, _, _ = jarque_bera(residuals['y3'])
jb_y8_stat, jb_y8_pval, _, _ = jarque_bera(residuals['y8'])

print("\nJarque-Bera Normality Test for Residuals:")
print(f"y3 Residuals: Statistic={jb_y3_stat:.3f}, p-value={jb_y3_pval:.3f}")
print(f"y8 Residuals: Statistic={jb_y8_stat:.3f}, p-value={jb_y8_pval:.3f}")

alpha_norm = 0.05
print(f"\nConclusion at alpha={alpha_norm}:")
if jb_y3_pval < alpha_norm:
    print(" - Reject normality for y3 residuals.")
else:
    print(" - Cannot reject normality for y3 residuals.")
if jb_y8_pval < alpha_norm:
    print(" - Reject normality for y8 residuals.")
else:
    print(" - Cannot reject normality for y8 residuals.")

# Multivariate Normality Test (using VECMResults method)
try:
    normality_test_results = vecm_results.test_normality()
    print("\nMultivariate Normality Test Results (using VECMResults method):")
    print(normality_test_results)
except Exception as e:
    print(f"\nCould not run multivariate normality test directly: {e}")

# %% [markdown]
# There is not enough evidence to reject the H0 saying that distributions are normal.
#
# ---

# %% [markdown]
# ## Forecasting using VECM
#
# We are going to calculate forecast for the next 25 periods using VECM model. We alraedy have data divided into sample (575 observations) and test (25 observations) sets. Because we conducted all of previous operations on in sample data already, we can proceed directly to forecasting.

# %%
forecast_periods = 25
# We want both point forecast and most probable range (within 95 % confidence interval)
forecast_values = vecm_results.predict(steps=forecast_periods, alpha=0.05)
# Creating dataframe with all data for further research
forecast_df = pd.DataFrame(forecast_values[0], index=coint_test_df.index,
                           columns=['y3_fore', 'y8_fore'])
lower_df = pd.DataFrame(forecast_values[1], index=coint_test_df.index,
                           columns=['y3_lower', 'y8_lower'])
upper_df = pd.DataFrame(forecast_values[2], index=coint_test_df.index,
                           columns=['y3_upper', 'y8_upper'])

# Combine forecasts and intervals
forecast_merged= pd.concat([forecast_df, lower_df, upper_df], axis=1)
print("\nForecasts and Confidence Intervals:")
print(forecast_merged)

# %%
# Merge forecasts with the original data for plotting
df_merged = df[['y3', 'y8']].merge(forecast_merged, left_index=True, right_index=True, how='left')
plot_forecast_with_ci('2025-03-06',df_merged,'y3','y3_fore','y3_lower','y3_upper','25 days Forecast vs Actual for y3','Price')

# %%
# Merge forecasts with the original data for plotting
# df_merged = df_original.merge(forecast_df, left_index=True, right_index=True, how='left')
plot_forecast_with_ci('2025-03-06',df_merged,'y8','y8_fore','y8_lower','y8_upper','25 days Forecast vs Actual for y8','Price')

# %% [markdown]
# Forecast is not close to the real prices of assets in neither of those cases. Actual values are within our 95% confidence interval though.
# In following section we will examine metrics for our out-of-sample period forecast errors.

# %%
print_accuracy_measures(df_merged.dropna(),4,'y3','y8')

# %% [markdown]
# ---

# %% [markdown]
# ## ARIMA
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



