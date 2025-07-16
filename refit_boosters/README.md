# Improve boosting models
Booster methods allows refit and update models option.

In this repository we explore some ways to take advantage of it.

### Requirements
* Python 3.11
* Poetry (e.g. ```brew install poetry```)

### Set up of the Python environment
* To get all the required packages, type: 

```poetry install ```

Set up your Python interpreter in your IDE to use the virtual environment you just created.

* Create a Jupyter kernel called `boosters`:

```
python -m ipykernel install --user --name=boosters
```

### Set up environment variables
* Make a copy of the file `env_example` and rename it as `.env`
* Add your kaggle credentials. Intructions to get them are [here](https://www.kaggle.com/docs/api/#getting-started-installation-&-authentication), in the section ***Authentication***.

### Entry points
