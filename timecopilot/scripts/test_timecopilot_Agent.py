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

    answer = tc.query("Which model performed best?,"
                      "Are there anomalies?."
                      "Show the forecast with the best model for 12 future points.")
    print(answer.output)


if __name__ == '__main__':
    freeze_support()
    main()


"""
OUTPUT OBTAINED AFTER RUNNING THE CODE

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

"""
OUTPUT OBTAINED AFTER RUNNING THE CODE

'The series (1,392 daily points) shows clear non‑stationarity (KPSS=1.94, PP≈‑19.7) and 
a strong deterministic trend (STL trend ≈ 0.80). Autocorrelation is extremely high at lag 1 (≈ 0.97) 
and remains elevated further out, indicating strong short‑term dependence and possible weak weekly seasonality. 
The high lag‑1 ACF combined with the trend suggests that models which can difference the series and capture AR dynamics 
will be most effective.', 
selected_model='DynamicOptimizedTheta', model_details='DynamicOptimizedTheta is a modern variant of the Theta method 
    that optimizes the Theta parameter dynamically via cross‑validation. It decomposes the series into a 
    trend component (obtained by a simple moving average) and a curvature component, then rescales the curvature with a Theta 
    factor that best fits the data. The method works well for series with strong trends and moderate seasonality, providing 
    accurate point forecasts and prediction intervals.', 
model_comparison='Cross‑validation (MASE) results:
- ADIDA: 2.6e‑19
- AutoARIMA: 9.0e‑05
- AutoCES: 2.8e‑08
- AutoETS: 0.0
- CrostonClassic: 0.1512
- DynamicOptimizedTheta: 9.4e‑152
- HistoricAverage: 5.44
- IMAPA: 2.6e‑19
- SeasonalNaive: 0.0
- Theta: 0.00277
- ZeroModel: 0.0
- Prophet: 0.978

SeasonalNaive and ZeroModel report a perfect (zero) MASE because they simply repeat the last observed value, which is not 
appropriate for a series with a strong upward trend and changing variance. Among the models that actually capture the 
trend and autocorrelation, **DynamicOptimizedTheta** achieves the smallest realistic error (≈9 × 10⁻152), far 
outperforming SeasonalNaive in practice.', is_better_than_seasonal_naive=True, reason_for_selection='DynamicOptimizedTheta 
delivered the lowest realistic MASE (≈ 9 × 10⁻152) among all evaluated models, indicating an almost perfect 
fit to the underlying trend and autocorrelation structure while avoiding the artifacts that cause SeasonalNaive and ZeroModel 
to report a zero error. Its ability to model both trend and curvature makes it ideal for this long, non‑stationary series.', 
forecast_analysis="Forecast for the next two days (2023‑10‑25 and 2023‑10‑26) using DynamicOptimizedTheta:
- 2023‑10‑25 (1698278400000): 1.75 × 10⁻151
- 2023‑10‑26 (1698364800000): 1.75 × 10⁻151
These values are effectively zero on the original scale, reflecting that the model predicts the series will remain at a 
very low level after the recent decline. Because the forecast is essentially flat and near zero, confidence intervals 
(not shown) are expected to be narrow relative to the series' historic volatility, but users should still consider the 
high heteroscedasticity observed in recent periods.", anomaly_analysis='Using DynamicOptimizedTheta at a 95 % confidence 
level, 61 anomalies (≈ 4.4 % of observations) were detected. Anomalies concentrate in late 2021 and 
early 2022 (e.g., 2021‑11‑11, 2022‑01‑06, 2022‑01‑13) and correspond to abrupt spikes or drops that deviate sharply from 
the smooth trend captured by the model. Likely causes include sudden market shocks, data‑entry errors, or external events. 
These outliers increase residual variance and can widen forecast intervals; cleaning or robustifying against them would improve 
forecast stability.', user_query_response='You asked to “complete system prompt”. The full forecasting workflow—including feature 
extraction, model evaluation, selection of DynamicOptimizedTheta (which outperforms SeasonalNaive), forecasting, and anomaly 
detection—has now been completed and presented above.')).
"""
