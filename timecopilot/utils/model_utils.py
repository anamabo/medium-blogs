from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot_results(results_df: pd.DataFrame, target_colname: str, title: str, figsize = (7, 6)):
    fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Change font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    cmap = plt.get_cmap('viridis')
    list_models = set(
        results_df.columns[results_df.columns != target_colname].str.replace(r'-(lo|hi)-90', '', regex=True)
    )
    colors = [cmap(i) for i in np.linspace(0, 1, len(list_models))]
    for model, color in zip(list_models, colors):
        ax.plot(results_df.index, results_df[model].values, color=color, linestyle="-", label=model)
        ax.fill_between(
            results_df.index,
            results_df[f"{model}-lo-90"].values,
            results_df[f"{model}-hi-90"].values,
            color=color,
            alpha=0.2,
            label=f"{model} prediction 90%",
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
