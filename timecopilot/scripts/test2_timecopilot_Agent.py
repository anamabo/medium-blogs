import os
import pandas as pd
from timecopilot import TimeCopilot
from multiprocessing import Process, freeze_support
import nest_asyncio

nest_asyncio.apply()


def main():
    """
    dataset2.csv contains a missing date, which causes timecopilot errors, here we show how to deal with it
    missing date 2020-03-06,14
    """
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "dataset2.csv"), parse_dates=True)
    date_colname= "date"
    target_colname = "target"
    df[date_colname] = pd.to_datetime(df[date_colname])
    df.rename(columns={date_colname: "ds", target_colname: "y"}, inplace=True)
    df = df.set_index('ds').resample('D').ffill().reset_index()
    df["unique_id"] = "dataset2"
    cols = ['unique_id', 'ds', 'y']

    limit_date = '2023-01-02'
    df_train = df[df['ds'] < limit_date].copy()
    df_test = df[df['ds'] >= limit_date].copy()

    tc = TimeCopilot(llm="<The LLM you selected>", retries=3)
    # listing all default models that timecopilot uses for forecasting and cross-validation.
    # DEFAULT_MODELS: list[Forecaster] = [
    #     ADIDA(), AutoARIMA(), AutoCES(), AutoETS(), CrostonClassic(), DynamicOptimizedTheta(),
    #     HistoricAverage(), IMAPA(), SeasonalNaive(), Theta(), ZeroModel(), Prophet(),
    # ]

    result = tc.forecast(df=df_train[cols])
    print(result.output.tsfeatures_analysis)

    answer = tc.query("Make a plot with the decomposition of the time_series.")
    print(answer.output)

    answer = tc.query("Which model performed best?")
    print(answer.output)

    answer = tc.query("Are there anomalies?, create a plot showing the anomalies.")
    print(answer.output)

    answer = tc.query("Show the forecast with the best model for 12 future points.")
    print(answer.output)


if __name__ == '__main__':
    freeze_support()
    main()


"""
OUTPUT OBTAINED AFTER RUNNING THE CODE

The time series feature analysis reveals significant insights about the dataset2:

- **Stationarity**: The KPSS test value is (3.68) > 0.05, indicating the series likely has a 
unit root and is non-stationary. Similarly, the PP test gives a strong value of (-16.36) indicating 
the series is likely non-stationary.
- **Trending Behavior**: The trend feature is 0.829 (high positive), suggesting the time series 
has an upward trend. However, the curvature is negative (-3.12), revealing a possible inflection 
or flattening in the trend.
- **Seasonality**: Seasonal period is set as 1, and no clear indication of high seasonality is 
detected.
- **Autocorrelation and PACF Patterns**: The ACF and PACF values are high, indicating moderate 
to strong autocorrelation at short lag intervals. This is a key consideration for models like 
ARIMA that rely on exploiting autocorrelation patterns.
- **Non-linearity and Complexity**: The nonlinearity feature has a moderate value (0.758), indicating 
the time series might benefit more from non-linear models or additional preprocessing.
- **Irregularity/Noise**: Entropy is moderate (0.4216), suggesting the series isn't too noisy, but 
some level of complexity or irregularity exists.
- **Crossing Points**: Relatively high (59 crossing points), which can indicate the time series is 
volatile or changes direction frequently.

These features suggest that an appropriate model must manage non-stationarity, detect and account for 
trends, and potentially handle moderate autocorrelation and non-linear structures in the series.

***Query: Make a plot with the decomposition of the time_series.
I'm unable to directly create a decomposition plot for the time series since that functionality is 
not included in the list of available tools. Let me know if there is anything else I can help you with!

***Query: Which model performed best?
To determine the best-performing model, we refer to the evaluation results provided in the analysis, 
which uses MASE (Mean Absolute Scaled Error) as the evaluation metric. The lower the MASE value, the 
better the model performance. Here are the MASE values for the models:

- **AutoARIMA**: 0.34923
- **AutoETS**: 0.63363
- **Prophet**: 1.35146
- **SeasonalNaive**: 0.63364
- **HistoricAverage**: 1.02093

Based on the MASE values, **AutoARIMA** performed best with the lowest MASE score of **0.34923**, 
indicating that it has the highest forecasting accuracy among the models.

***Query: Are there anomalies?, create a plot showing the anomalies.
Yes, there are anomalies in the dataset. According to the analysis:

- The AutoARIMA model detected 50 anomalies in the time series "dataset2"
- This represents 4.57% of the total data points (50 out of 1094)
- The first few detected anomaly dates include: 
January 6, 2022; January 13, 2022; January 20, 2022; January 22, 2022; January 24, 2022; 
January 27, 2022; January 31, 2022; February 3, 2022; February 4, 2022; and February 5, 2022

The plot I've generated shows the time series with anomalies highlighted, making it easy
to identify where these unusual data points occur in relation to the overall pattern. The 
anomalies represent points where the actual values deviated significantly from what the model 
expected based on the established patterns in the data.

***Query: Show the forecast with the best model for 12 future points.

I've generated a forecast plot using the best-performing model (AutoARIMA) for 12 future time 
points. The plot shows:

1. Historical data (actual values) in blue
2. Forecasted values in orange 
3. Confidence intervals as a shaded area around the forecast

The AutoARIMA model, which had the lowest MASE score of 0.349, predicts the future values of the 
time series "dataset2". 
The forecast covers 12 future periods, and you can see the predicted trajectory continuing from 
the recent trend in the historical data.

The confidence intervals (shaded area) represent the uncertainty around the forecast. As expected, 
the uncertainty generally increases 
with the forecast horizon, reflecting greater unpredictability for longer-term predictions.

The forecasted values suggest that the time series will continue at a similar level to the most 
recent observations, with the model capturing the underlying patterns and trends in the data.
"""
