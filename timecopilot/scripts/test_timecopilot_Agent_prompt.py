import os
import pandas as pd
from timecopilot import TimeCopilot
from pydantic_ai.models.bedrock import BedrockConverseModel

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.ollama import OllamaProvider
from multiprocessing import Process, freeze_support
import time

from timecopilot.models.prophet import Prophet
# statsmodels
from timecopilot.models.stats import ADIDA, AutoARIMA, AutoCES, AutoETS, CrostonClassic, DynamicOptimizedTheta, HistoricAverage, IMAPA, SeasonalNaive, Theta, ZeroModel
# foundation models
from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.foundation.flowstate import FlowState
from timecopilot.models.foundation.moirai import Moirai
from timecopilot.models.foundation.sundial import Sundial
from timecopilot.models.foundation.tabpfn import TabPFN
from timecopilot.models.foundation.tirex import TiRex
from timecopilot.models.foundation.timegpt import TimeGPT
from timecopilot.models.foundation.timesfm import TimesFM
from timecopilot.models.foundation.toto import Toto
# ml models
from timecopilot.models.ml import AutoLGBM
# nn models
from timecopilot.models.neural import AutoNHITS, AutoTFT

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

    user_query = 'Please do a decomposition and separate the trend and seasonality. Please also provide the confidence intervals for the forecast. Please also provide an explanation of the forecast.'

    st = time.time()
    tc = TimeCopilot(llm="The_LLM_you_selected", retries=3)
    # listing all default models that timecopilot uses for forecasting and cross-validation.
    # DEFAULT_MODELS: list[Forecaster] = [
    #     ADIDA(), AutoARIMA(), AutoCES(), AutoETS(), CrostonClassic(), DynamicOptimizedTheta(),
    #     HistoricAverage(), IMAPA(), SeasonalNaive(), Theta(), ZeroModel(), Prophet(),
    # ]

    result = tc.forecast(df=df_train[cols], query=user_query)
    print(f"Run Time: {time.time() - st}")
    print(result.output.tsfeatures_analysis)

    answer = tc.query("Make a plot with the decomposition of the forecast and the confidence intervals.")
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

The time series features reveal several important characteristics about the data:

1. Strong trend presence (trend: 0.8298) - indicating a clear upward movement over time
2. High stability (stability: 0.9509) - the series maintains consistent statistical properties
3. High autocorrelation at lag 1 (x_acf1: 0.9686) and lag 10 (x_acf10: 8.4312) - indicating strong persistence in the data
4. Non-stationary nature (unitroot_pp: -16.3646) - the series has a unit root and is not stationary
5. Length: 1095 observations representing approximately 3 years of daily data
6. Low lumpiness (0.0477) and moderate entropy (0.4216) - suggesting relatively smooth transitions
7. No seasonal period detected (seasonal_period: 1) - indicating no clear seasonal pattern

The series shows a strong trend component with high persistence but no clear seasonality. The non-stationarity suggests that 
differencing or trend modeling will be important for accurate forecasting. The high stability indicates that the underlying 
data generation process is relatively consistent over time.

*** Make a plot with the decomposition of the forecast and the confidence intervals.
I apologize, but I don't have a specific tool available to create a plot with the decomposition of the forecast and confidence intervals. 
The plot_tool function we have doesn't directly support this type of visualization. 

However, I can offer you some alternative visualizations that might be helpful:

1. We can show you the forecast plot, which will display the actual values and the forecasted values. This might give you some insight into the model's performance.

2. We can also show you the raw time series data, which could help you understand the overall pattern and any potential seasonality or trends.


***Which model performed best?
The AutoARIMA model performed best with a MASE (Mean Absolute Scaled Error) score of 0.349. This is significantly better than the other models:

- AutoARIMA: 0.349
- DynamicOptimizedTheta: 0.625
- AutoETS: 0.634
- SeasonalNaive: 0.634
- Theta: 0.635
- Prophet: 1.351

The lower the MASE score, the better the model performance, with values below 1.0 indicating better performance than a naive seasonal forecast. 
AutoARIMA's score of 0.349 suggests it captures the time series patterns significantly better than the other models.

So, yes – the series contains anomalous points.

---
***Are there anomalies?, create a plot showing the anomalies.
Yes, there are anomalies in the dataset. According to the analysis:

- The AutoARIMA model detected 50 anomalies in the time series "dataset2"
- This represents 4.57% of the total data points (50 out of 1094)
- The first few detected anomaly dates include: 
January 6, 2022; January 13, 2022; January 20, 2022; January 22, 2022; January 24, 2022; January 27, 2022; January 31, 2022; February 3, 2022; February 4, 2022; and February 5, 2022

The plot I've generated shows the time series with anomalies highlighted, making it easy to identify where these unusual data points occur in relation 
to the overall pattern. The anomalies represent points where the actual values deviated significantly from what the model expected based on the 
established patterns in the data.

---

I've generated a forecast plot using the best-performing model (AutoARIMA) for 12 future time points. The plot shows:

1. Historical data (actual values) in blue
2. Forecasted values in orange 
3. Confidence intervals as a shaded area around the forecast

The AutoARIMA model, which had the lowest MASE score of 0.349, predicts the future values of the time series "dataset2". 
The forecast covers 12 future periods, and you can see the predicted trajectory continuing from the recent trend in the historical data.

The confidence intervals (shaded area) represent the uncertainty around the forecast. As expected, the uncertainty generally increases 
with the forecast horizon, reflecting greater unpredictability for longer-term predictions.

The forecasted values suggest that the time series will continue at a similar level to the most recent observations, with the model 
capturing the underlying patterns and trends in the data.
"""
