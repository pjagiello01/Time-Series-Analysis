import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from typing import Type

def adf_test(series, max_aug=10, version='n'):
    
    results = []

    y = series.diff()
    X = pd.DataFrame({'y_lag': series.shift()})

    if version == 'c' or version == 't': # constant to be added optionally 
        X = sm.add_constant(X)
    if version == 't': # (deterministic) trend component to be added optionally
        X['trend'] = range(len(X))

    for i in range(0, max_aug): # iterating through different numbers of augmentations
        
        for aug in range(1, i+1): # adding augmentations one by one until its current amount is reached
            X['aug_'+str(aug)] = y.shift(aug)

        model = sm.OLS(series.diff(), X, missing='drop').fit() # fitting a linear regression with OLS

        ts = model.tvalues['y_lag'] # test statistic
        nobs = model.nobs # number of observations

        if version == 'n': # critical values for basic version of ADF
            if nobs > 500:
                cv1 = -2.567; cv5 = -1.941; cv10 = -1.616 # critical values for more than 500 observations
            else:
                cv1 = np.nan; cv5 = np.nan; cv10 = np.nan # if number of observations is lower than 500, we should check the critical values manually
        if version == 'c': # critical values for version with constant
            if nobs > 500:
                cv1 = -3.434; cv5 = -2.863; cv10 = -2.568 # critical values for more than 500 observations
            else:
                cv1 = np.nan; cv5 = np.nan; cv10 = np.nan # if number of observations is lower than 500, we should check the critical values manually
        if version == 't': # critical values for version with constant and (deterministic) trend component
            if nobs > 500:
                cv1 = -3.963; cv5 = -3.413; cv10 = -3.128 # critical values for more than 500 observations
            else:
                cv1 = np.nan; cv5 = np.nan; cv10 = np.nan # if number of observations is lower than 500, we should check the critical values manually

        bg_test5 = smd.acorr_breusch_godfrey(model, nlags=5); bg_pvalue5 = round(bg_test5[1],4)
        bg_test5 = smd.acorr_breusch_godfrey(model, nlags=10); bg_pvalue10 = round(bg_test5[1],4)
        bg_test5 = smd.acorr_breusch_godfrey(model, nlags=15); bg_pvalue15 = round(bg_test5[1],4)

        results.append([i, ts, cv1, cv5, cv10, bg_pvalue5, bg_pvalue10, bg_pvalue15])

    results_df = pd.DataFrame(results)
    results_df.columns = ['number of augmentations', 'ADF test statistic', 'ADF critival value (1%)', 'ADF critival value (5%)', 'ADF critival value (10%)', 'BG test (5 lags) (p-value)', 'BG test (10 lags) (p-value)', 'BG test (15 lags) (p-value)']
    
    return results_df


def get_forecast_df(n_steps: int,
                    model: Type[ARIMA]):
    forecast_results = model.get_forecast(steps=n_steps)
    mean_forecast = forecast_results.predicted_mean
    conf_intervals = forecast_results.conf_int()
    forecast_df = pd.DataFrame({
        "forecast": mean_forecast,
        "lower ci": conf_intervals[:, 0],
        "upper ci": conf_intervals[:, 1]
    })

    return forecast_df




def plot_forecast_with_ci(start_date: str,
                          df,
                          actual_col: str,
                          forecast_col: str,
                          lower_col: str,
                          upper_col: str,
                          title: str,
                          ylabel: str = 'Value'):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual values
    df.loc[start_date:, actual_col].plot(ax=ax, label='Actual', color='black', marker='.')

    # Plot forecasted values
    df.loc[start_date:, forecast_col].plot(ax=ax, label='Forecast', color='blue', linestyle='--')

    # Plot confidence interval
    ax.fill_between(df.loc[start_date:].index,
                    df.loc[start_date:, lower_col],
                    df.loc[start_date:, upper_col],
                    color='red', alpha=0.2, label='95% CI')

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()


# Calculate forecast accuracy measures
def mape(actual, pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual - pred) / actual)) * 100


def amape(actual, pred):
    """Adjusted/Symmetric Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual - pred) / ((actual + pred) / 2))) * 100


def calculate_accuracy_measures(forecast_evaluation, variable):
    # Calculate metrics for variable
    first_variable_mae = mean_absolute_error(forecast_evaluation[f'{variable}'],
                                             forecast_evaluation[f'{variable}_fore'])
    first_variable_mse = mean_squared_error(forecast_evaluation[f'{variable}'],
                                            forecast_evaluation[f'{variable}_fore'])
    first_variable_rmse = np.sqrt(first_variable_mse)
    first_variable_mape = mape(forecast_evaluation[f'{variable}'], forecast_evaluation[f'{variable}_fore'])
    first_variable_amape = amape(forecast_evaluation[f'{variable}'],
                                 forecast_evaluation[f'{variable}_fore'])
    return first_variable_mae, first_variable_mse, first_variable_rmse, first_variable_mape, first_variable_amape


def print_accuracy_measures(forecast_evaluation, lags, first_variable, second_variable):
    # Create a DataFrame to display the results
    metrics_df = pd.DataFrame({
        f'{first_variable}': calculate_accuracy_measures(forecast_evaluation, first_variable),
        f'{second_variable}': calculate_accuracy_measures(forecast_evaluation, second_variable),
    }, index=['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'AMAPE (%)'])
    print(f"Forecast Accuracy Metrics for {lags} lags:")
    print(metrics_df)