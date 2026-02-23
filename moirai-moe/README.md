# Moirai-MoE: Time Series Foundation Models with Sparce Mixture of Experts

Welcome to this repository! Here you find the codes related to the blog post written in medium:
[TBD]().

### Requirements
* Python 3.12
* Poetry (e.g. ```brew install poetry```)

### Set up of the Python environment
* To get all the required packages, type: 

```poetry install ```

Set up your Python interpreter in your IDE to use the virtual environment you just created.

* Create a Jupyter kernel called ```moirae-moe``` to run the notebooks:

```python -m ipykernel install --user --name=moirae-moe```

### Structure of the repository
* ```notebooks```: contains the notebooks to run the experiments.
  * [explore_data.ipynb](notebooks/explore_data.ipynb): Used to get the raw data, select and preprocess the time series data.
  * [moiraimoe_forecast_datasetX.ipynb](notebooks/moiraimoe_forecast_dataset1.ipynb): It trains Moirai-MoE in the datasetX (X=1,2,3). Here you find a functionality to do rolling forecast.
* ```scripts```: contains functions to analyze, model and plot time series data.
* ```data```: It contains the datasets used to experiment with Moirai-MoE. You can also obtain these 
datasets by running ```notebooks/explore_data.ipynb```.

