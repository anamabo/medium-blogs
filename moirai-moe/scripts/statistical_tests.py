from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from typing import Tuple


def seasonal_decompose_plot(ts: pd.DataFrame, colname: str, period: int, model: str) -> DecomposeResult:
    # Decompose the time series into trend, seasonal and residual components
    # model can be "additive" or "multiplicative"
    result = seasonal_decompose(ts[colname], model=model, period=period)
    result.plot()
    plt.show()

    return result


def plot_auto_corr(ts: pd.DataFrame, colname: str, lags: int):
    # auto-correlation plot
    plot_acf(ts[colname], lags=lags)
    plt.show()


def test_autocorrelation(ts: pd.DataFrame, colname: str, lags: int) -> Tuple[float, float]:
    """
    The Ljung–Box test examines whether there are significant autocorrelations in a time series.
    H0: The data is not correlated -> any observed corrs are due to randomness.
    Ha: The data is correlated -> the observed corrs are not due to randomness.
    """
    results = acorr_ljungbox(ts[colname], lags=[lags])
    lb_stat = results["lb_stat"].values[0]
    pval = results["lb_pvalue"].values[0]
    if pval < 0.05:
        print(f"Reject H0, thus the data is correlated. Ljung-Box statistic: {lb_stat}, p-value: {pval}")
    else:
        print(f"Accept H0, thus the data is not correlated. Ljung-Box statistic: {lb_stat}, p-value: {pval}")

    return lb_stat, pval


def test_stationarity(ts: pd.DataFrame, colname: str):
    """ Augmented Dickey-Fuller test for stationarity
        H0= TS is not stationary => if statistics > crit vals, we can not reject Ho (left test).
    """
    result = adfuller(ts[colname])
    adf_statistic, p_value, used_lag, n_obs, critical_values, icbest = result

    if p_value < 0.05:
        print(f"Reject H0, thus the data is stationary. ADF statistic: {adf_statistic}, p-value: {p_value}")
    else:
        print(f"Accept H0, thus the data is not stationary. ADF statistic: {adf_statistic}, p-value: {p_value}")

    return result


def explore_timeseries(table: pd.DataFrame, ycolname: str, period: int = 365, model: str = "additive"):
    print("Decomposition:")
    _ = seasonal_decompose_plot(table, colname=ycolname, period=period, model=model)

    print("Auto correlation test:")
    _, _ = test_autocorrelation(table, colname=ycolname, lags=period)

    print("Stationarity test:")
    _ = test_stationarity(table, colname=ycolname)
