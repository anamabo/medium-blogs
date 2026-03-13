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

    user_prompt = (
        "Do a decomposition of the time-series data and separate the trend and seasonality."
        "Provide a forecast for a horizon of 90 days with the respective 90% confidence intervals."
        "Provide an explanation of the forecast."
    )

    tc = TimeCopilot(llm="<The LLM you selected>", retries=3)
    # listing all default models that timecopilot uses for forecasting and cross-validation.
    # DEFAULT_MODELS: list[Forecaster] = [
    #     ADIDA(), AutoARIMA(), AutoCES(), AutoETS(), CrostonClassic(), DynamicOptimizedTheta(),
    #     HistoricAverage(), IMAPA(), SeasonalNaive(), Theta(), ZeroModel(), Prophet(),
    # ]

    result = tc.forecast(df=df_train[cols], query=user_prompt)
    print(result.output.tsfeatures_analysis)


if __name__ == '__main__':
    freeze_support()
    main()


"""
OUTPUT OBTAINED AFTER RUNNING THE CODE

The time series features reveal several important characteristics about the data:

1. Strong trend presence (trend: 0.8298) - indicating a clear upward movement over time
2. High stability (stability: 0.9509) - the series maintains consistent statistical properties
3. High autocorrelation at lag 1 (x_acf1: 0.9686) and lag 10 (x_acf10: 8.4312) - indicating strong persistence in the data
4. Non-stationary nature (unitroot_pp: -16.3646) - the series has a unit root and is not stationary
5. Length: 1095 observations representing approximately 3 years of daily data
6. Low lumpiness (0.0477) and moderate entropy (0.4216) - suggesting relatively smooth transitions
7. No seasonal period detected (seasonal_period: 1) - indicating no clear seasonal pattern
8. The STL (Seasonal and Trend decomposition) features aren't available in this analysis.

The series shows a strong trend component with high persistence but no clear seasonality. The non-stationarity suggests that 
differencing or trend modeling will be important for accurate forecasting. The high stability indicates that the underlying 
data generation process is relatively consistent over time.

"""
