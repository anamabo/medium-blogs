from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot_results(results_df: pd.DataFrame, target_colname: str, title: str, figsize = (7, 6)):
    fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Change font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    ax.plot(results_df.index, results_df["AutoARIMA"].values, "g-", label="AutoARIMA")
    ax.fill_between(
        results_df.index,
        results_df["AutoARIMA-lo-90"].values,
        results_df["AutoARIMA-hi-90"].values,
        color="g",
        alpha=0.2,
        label="AutoARIMA prediction 90%",
    )

    ax.plot(results_df.index, results_df["Moirai"].values, "r-", label="Moirai")
    ax.fill_between(
        results_df.index,
        results_df["Moirai-lo-90"].values,
        results_df["Moirai-hi-90"].values,
        color="r",
        alpha=0.2,
        label="Moirai prediction 90%",
    )

    ax.plot(results_df.index, results_df["Prophet"].values, "c-", label="Prophet")
    ax.fill_between(
        results_df.index,
        results_df["Prophet-lo-90"].values,
        results_df["Prophet-hi-90"].values,
        color="c",
        alpha=0.2,
        label="Prophet prediction 90%",
    )
    ax.plot(results_df.index, results_df[target_colname].values, "b-", label=target_colname)

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
