from matplotlib import pyplot as plt
import pandas as pd


def plot_time_series(table: pd.DataFrame, title: str, figsize=(8, 6)):
    fig = plt.figure(figsize=figsize)

    plt.plot(table["date"].values, table["target"].values)
    plt.xlabel('date', size=16)
    plt.ylabel('Value', size=16)

    # Change font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # Rotate x-tick labels
    plt.xticks(rotation=45)

    # Make plot's frame invisible
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    # Add a grid for better newspaper aesthetics
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.title(title, fontsize=14)
    plt.tight_layout()
    return fig