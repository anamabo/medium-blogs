import os
from timecopilot import TimeCopilotForecaster
# prophet
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
import time
import pandas as pd
from multiprocessing import Process, freeze_support
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse

import matplotlib.pyplot as plt


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

    st = time.time()
    tcf = TimeCopilotForecaster(
        models=[
            # statistical models
            ADIDA(), AutoARIMA(), AutoCES(), AutoETS(), CrostonClassic(), DynamicOptimizedTheta(), HistoricAverage(), IMAPA(), SeasonalNaive(), Theta(), ZeroModel(),
            # foundation models
            Chronos(), Moirai(), Sundial(), TiRex(), TimesFM(), Toto(),  # FlowState(), TabPFN(), TimeGPT(),
            # ml models
            AutoLGBM(),  # does not support quantiles
            # nn models
            AutoNHITS(), AutoTFT(),
            # prophet
            Prophet(),
        ]
    )
    cv_results = tcf.cross_validation(
        df=df_train[cols],
        h=180,          # Forecast horizon
        n_windows=3     # Number of CV folds
    )

    eval_df = evaluate(
        cv_results.drop(columns=["cutoff"]),
        metrics=[mae, rmse],
    )
    print(time.time() - st)
    eval_df.to_csv('forecaster_eval.csv')

    fcst_df = tcf.forecast(df=df_train, h=180, level=[90])
    fcst_df.to_csv('forecaster_forecast.csv', index=False)

    fig = tcf.plot(df, fcst_df)
    fig.savefig("fig_forecaster_forecast.png")


if __name__ == '__main__':
    freeze_support()
    main()
