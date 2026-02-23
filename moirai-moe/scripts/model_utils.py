from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch


def generate_model_input(train_df: pd.DataFrame, date_colname: str, target_colname: str, median_forecast: np.array = None):
    train_ts = train_df.copy()

    if median_forecast is not None:
        last_date_train = train_ts.index[-1]
        frequency = train_ts.index.diff()[1]
        median_forecast_ts = pd.DataFrame(
            {
                date_colname: pd.date_range(last_date_train+ frequency, periods=len(median_forecast), freq=frequency),
                target_colname: median_forecast,
            }
        )

        median_forecast_ts.set_index(date_colname, inplace=True)
        # concatenate the train and the forecast
        train_ts = pd.concat([train_ts, median_forecast_ts], axis=0)

    return train_ts


def preprocess_data(target: np.array):
    # NOTE!!!!! This function assumes that  the time series has No null values and NO padding values.
    # If there are null values, please, clean your TS first.

    # 1. Reshape target values. Shape: (batch, time, variate)
    # For 1D time series, batch =1, time = len(target), variate = 1
    tensor_target = rearrange(
        torch.as_tensor(target, dtype=torch.float32), "t -> 1 t 1"
    )
    #2. Create tensor whether there is a value or not.
    # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
    past_observed_target = torch.ones_like(tensor_target, dtype=torch.bool)
    # 3. Tensor to say whether a value is padded or not.
    # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
    past_padded_target = torch.zeros_like(tensor_target, dtype=torch.bool).squeeze(-1)
    return tensor_target, past_observed_target, past_padded_target


def get_predictions(forecast_tensor: torch.Tensor, train_set: pd.DataFrame, date_colname: str, percentile_inf: int = 5, percentile_sup: int = 95):
    """
    Get the median and the prediction interval of the forecast.
    Args:
        forecast_tensor (torch.Tensor): The forecast tensor. Shape=(num_samples, prediction_window)
        percentile_inf (int): The lower percentile of the prediction interval.
        percentile_sup (int): The upper percentile of the prediction interval.
    """
    median_prediction = np.round(np.median(forecast_tensor, axis=0), decimals=4)
    inf_prediction = np.percentile(forecast_tensor, percentile_inf, axis=0)
    sup_prediction = np.percentile(forecast_tensor, percentile_sup, axis=0)

    # Add these info into a DataFrame
    last_date_train = train_set.index[-1]
    frequency = train_set.index.diff()[1]
    all_forecast_ts = pd.DataFrame(
        {
            date_colname: pd.date_range(last_date_train+ frequency, periods=len(median_prediction), freq=frequency),
            "median_forecast": median_prediction,
            "inf_forecast": inf_prediction,
            "sup_forecast": sup_prediction,
        }
    )
    all_forecast_ts.set_index(date_colname, inplace=True)

    return all_forecast_ts


def plot_results(results_df: pd.DataFrame, target_colname: str, title: str, figsize = (7, 6)):
    fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Change font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    ax.plot(results_df.index, results_df["median_forecast"].values, "g-", label="median forecast")
    ax.plot(results_df.index, results_df[target_colname].values, "b-", label=target_colname)
    ax.fill_between(
        results_df.index,
        results_df["inf_forecast"].values,
        results_df["sup_forecast"].values,
        color="g",
        alpha=0.2,
        label="prediction 90%",
    )
    ax.set_xlabel('date', size=16)
    ax.set_ylabel('Value', size=16)
    ax.set_title(title, size=16)
    ax.legend()
    plt.xticks(rotation=45, ha='right')

    # Make plot's frame invisible
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    # Add a grid for better newspaper aesthetics
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Improve spacing and aspect ratio
    plt.tight_layout()  # Adjusts the plot to ensure labels don't get cut off
    plt.subplots_adjust(bottom=0.2)  # Add more space at the bottom for rotated labels

    # Optional: Set aspect ratio explicitly if needed
    # ax.set_box_aspect(0.6)  # For matplotlib >= 3.3.0 (width:height ratio)
    return fig1


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)/ y_true)) * 100
