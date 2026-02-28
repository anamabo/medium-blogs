import os
import pandas as pd
from timecopilot import TimeCopilot
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.ollama import OllamaProvider
from multiprocessing import Process, freeze_support

# sktime
from sktime.forecasting.trend import TrendForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

import nest_asyncio

nest_asyncio.apply()


def main():
    """
    Forecasters from sktime can be easily called from timecopilot.
    """
    df = pd.read_csv(os.path.join(os.getcwd(), "timecopilot", "data", "dataset3.csv"), parse_dates=True) 
    date_colname= "date"
    target_colname = "target"
    df[date_colname] = pd.to_datetime(df[date_colname])
    df["unique_id"] = "dataset3"
    df.rename(columns={date_colname: "ds", target_colname: "y"}, inplace=True)

    trend_forecaster = TrendForecaster()
    exp_smoothing = ExponentialSmoothing()
    tc = TimeCopilot(llm="<The LLM you selected>", 
                     forecasters=[trend_forecaster, exp_smoothing],
                     retries=3,)

    result = tc.forecast(df=df)
    print(result)

    answer = tc.query("Which is the best model?")
    print(answer.output)
    print(result)


if __name__ == '__main__':
    freeze_support()
    main()
