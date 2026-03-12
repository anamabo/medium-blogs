import os
from timecopilot import TimeCopilotForecaster

# sktime
from sktime.forecasting.trend import TrendForecaster

import time
import numpy as np
import pandas as pd
from multiprocessing import Process, freeze_support
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse

import matplotlib.pyplot as plt
from timecopilot.models.utils.forecaster import Forecaster


class Add_TrendForecaster(Forecaster):
    """Wrapper of TrendForecaster for timecopilot.
    Notes:
        - Level and quantiles are not supported for TrendForecaster yet. 
    """

    def __init__(
        self,
        alias: str = "TrendForecaster_alias",
        num_samples: int = 10,
        cv_n_windows: int = 5,
    ):
        self.alias = alias
        self.num_samples = num_samples
        self.cv_n_windows = cv_n_windows

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for time series data using the model.

        This method produces point forecasts. The input DataFrame can contain one
        or multiple time series in stacked (long) format.

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series to forecast. It must
                include as columns:

                    - "unique_id": an ID column to distinguish multiple series.
                    - "ds": a time column indicating timestamps or periods.
                    - "y": a target column with the observed values.

            h (int):
                Forecast horizon specifying how many future steps to predict.

        Returns:
            pd.DataFrame:
                DataFrame containing forecast results. Includes:

                    - point forecasts for each timestamp and series.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.
        """
        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for TrendForecaster yet.")

        freq = self._maybe_infer_freq(df, freq)
        tf = TrendForecaster(
            regressor=None
        )
        tf.fit(
            y=df['y'],
            fh=h,
        )
        fcst_df = pd.DataFrame()
        fcst_df[self.alias] = tf.predict(fh=np.arange(1, h + 1))
        fcst_df['ds'] = pd.date_range(start=df['ds'].max() + pd.Timedelta(1, unit=freq), periods=h, freq=freq)
        fcst_df['unique_id'] = df['unique_id'].iloc[0]
        return fcst_df


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

    trend_forecaster = Add_TrendForecaster()

    available_models = [trend_forecaster]

    st = time.time()
    tcf = TimeCopilotForecaster(
        models=available_models
    )
    cv_results = tcf.cross_validation(
        df=df_train[cols],
        h=180,          # Forecast horizon
        n_windows=3     # Number of CV folds
    )
    print(time.time() - st)

    eval_df = evaluate(
        cv_results.drop(columns=["cutoff"]),
        metrics=[mae, rmse],
    )
    eval_df.to_csv('forecaster_cv_eval.csv')

    fcst_df = tcf.forecast(df=df_train, h=180)
    fcst_df.to_csv('forecaster_forecast.csv', index=False)


if __name__ == '__main__':
    freeze_support()
    main()
