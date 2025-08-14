# Incremental learning in boosting methods
Welcome to the repository of incremental learning in boosting methods. Here you find the code 
corresponding to the blog post: [Incremental learning in LightGBM and XGBoost](https://medium.com/data-science-collective/incremental-learning-in-lightgbm-and-xgboost-9641c2e68d4b). 


### Requirements
* Python 3.11
* Poetry (e.g. ```brew install poetry```)

### Set up of the Python environment
To create an environment with all the required packages, type: 

```poetry install ```

Set up your Python interpreter in your IDE to use the virtual environment you just created.

To run the Jupyter Notebooks, you need to create a Jupyter kernel called `boosters`:

```
python -m ipykernel install --user --name=boosters
```


### Entry points
In the folder `notebooks/` you find implementations of incremental learning for classification and regression 
for both LightGBM and XGBoost. 
