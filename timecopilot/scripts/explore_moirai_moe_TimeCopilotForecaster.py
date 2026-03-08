import os
import sys
import time
from lightgbm import train
import pandas as pd
from timecopilot import TimeCopilotForecaster
from multiprocessing import Process, freeze_support
from timecopilot.models.foundation.moirai import Moirai


def main():
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "dataset3.csv"), parse_dates=True) 
    date_colname= "date"
    target_colname = "target"
    df[date_colname] = pd.to_datetime(df[date_colname])
    df["unique_id"] = "dataset3"
    df.rename(columns={date_colname: "ds", target_colname: "y"}, inplace=True)

    limit_date = '2014-06-01'
    df_train = df[df['ds'] < limit_date].copy()
    df_test = df[df['ds'] >= limit_date].copy()

    Moirai_Moe = Moirai(
        repo_id="Salesforce/moirai-moe-1.0-R-small",
        context_length=96,
        patch_size=16,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    st = time.time()
    tcf = TimeCopilotForecaster(
        models=[Moirai_Moe]
    )
    print(time.time() - st)
    fcst_df = tcf.forecast(df=df_train, h=12, level=[80])

    print(fcst_df)


if __name__ == '__main__':
    freeze_support()
    main()
