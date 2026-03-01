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
    df["unique_id"] = "dataset2"
    df.rename(columns={date_colname: "ds", target_colname: "y"}, inplace=True)

    df_left = pd.DataFrame()
    df_left['ds'] = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
    cols = ['unique_id', 'ds', 'y']

    df = pd.merge(df_left, df, how='left', on='ds')
    df['y'] = df['y'].fillna(0)
    df['unique_id'] = 'all'

    df_train = df[df['ds'] < '2023-01-02'].copy()
    df_test = df[df['ds'] >= '2023-01-02'].copy()

    st = time.time()
    tc = TimeCopilot(llm="The LLM you selected", retries=3)
    # listing all default models that timecopilot uses for forecasting and cross-validation.
    # DEFAULT_MODELS: list[Forecaster] = [
    #     ADIDA(), AutoARIMA(), AutoCES(), AutoETS(), CrostonClassic(), DynamicOptimizedTheta(),
    #     HistoricAverage(), IMAPA(), SeasonalNaive(), Theta(), ZeroModel(), Prophet(),
    # ]

    result = tc.forecast(df=df_train)
    print(f"Run Time: {time.time() - st}")
    print(result.output.tsfeatures_analysis)

    answer = tc.query("Which model performed best?, Are there anomalies?. Show the forecast with the best model for 12 future points.")
    print(answer.output)

    answer = tc.query("Are there anomalies?")
    print(answer.output)

    answer = tc.query("Show the forecast with the best model for 12 future points.")
    print(answer.output)


if __name__ == '__main__':
    freeze_support()
    main()


"""
OUTPUT OBTAINED AFTER RUNNING THE CODE

Run Time: 29.149405479431152
Key feature take‑aways:
- series_length = 1392 (daily data over ~3.8 years).
- flat_spots = 346, indicating many consecutive zeroes at the start and at the end.
- lumpiness = 0.055 (low), suggesting variance is relatively stable across time.
- entropy = 0.43 (moderate), showing some randomness but not extreme.
- unitroot_pp = –19.68 (strongly stationary according to PP), yet KPSS = 1.94 (non‑stationary), pointing to a deterministic trend 
rather than a stochastic one.
- Holt alpha ≈0.95, beta≈0, indicating a strong level component with little trend adjustment needed.
- No significant seasonal period detected (seasonal_period=1).
These characteristics favour models that handle trend and level shifts (ETS, ARIMA with differencing, Prophet) and make pure 
seasonal models less useful.

Which model performed best?
**Best‑performing model:** **DynamicOptimizedTheta**

**Why?**  
The evaluation table (`eval_df`) reports the Mean Absolute Scaled Error (MASE) for each candidate model:

| Model                     | MASE (lower = better) |
|---------------------------|-----------------------|
| AutoARIMA                 | 9.00 × 10⁻⁵ |
| AutoETS                   | **0.0** |
| Prophet                   | 0.978 |
| Theta                     | 2.77 × 10⁻³ |
| SeasonalNaive             | **0.0** |
| HistoricAverage           | 5.44 |
| **DynamicOptimizedTheta** | **9.35 × 10⁻¹⁵²** |

All models with a MASE of 0.0 (AutoETS, SeasonalNaive) are perfect on the test set, but **DynamicOptimizedTheta** achieves an even smaller
error—practically zero (9.35 × 10⁻¹⁵²). Because MASE is a scale‑free metric where lower values indicate better predictive accuracy, the 
DynamicOptimizedTheta model is the clear winner.

If you’d like to see the forecasts it produced (the two future points in `fcst_df`) or visualize the series and anomalies, just let me know!

So, yes – the series contains anomalous points.

---

The anomaly detection run for the AutoCES model found **61** outlier points, which is **4.39 %** of the 1 391 observations.  
A few of the anomaly dates are:

- 2021‑11‑11  
- 2021‑11‑18  
- 2022‑01‑06  
- 2022‑01‑13  
- 2022‑01‑20  

(Full list is in the `anomalies_df`.)

So, yes – the series contains anomalous points.

---

### 12‑step forecast from the best model (AutoCES)

| Forecast horizon (date) | Predicted value |
|--------------------------|-----------------|
| 2023‑10‑25 | 0.00003399 |
| 2023‑10‑26 | –0.00000126 |
| 2023‑10‑27 | 0.00002511 |
| 2023‑10‑28 | 0.00000408 |
| 2023‑10‑29 | 0.00001955 |
| 2023‑10‑30 | 0.00000696 |
| 2023‑10‑31 | 0.00001597 |
| 2023‑11‑01 | 0.00000838 |
| 2023‑11‑02 | 0.00001359 |
| 2023‑11‑03 | 0.00000897 |
| 2023‑11‑04 | 0.00001193 |
| 2023‑11‑05 | 0.00000908 |
"""
