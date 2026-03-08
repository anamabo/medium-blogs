import os
import pandas as pd
from timecopilot import TimeCopilot
from multiprocessing import Process, freeze_support

# fundational models within timecopilot
from timecopilot.models.foundation.moirai import Moirai
from timecopilot.models.foundation.chronos import Chronos

import nest_asyncio

nest_asyncio.apply()


def main():
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

    Moirai_Moe = Moirai(
        repo_id="Salesforce/moirai-moe-1.0-R-small",
        context_length=96,
        patch_size=16,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    tc = TimeCopilot(llm="<The LLM you selected>", forecasters=[Chronos(), Moirai_Moe], retries=3,)

    result = tc.forecast(df=df_train[cols])
    print(result)
    print(result.output.tsfeatures_analysis)

    answer = tc.query("Which is the best model?")
    print(answer.output)
    print(result)


if __name__ == '__main__':
    freeze_support()
    main()
