# SegmentWater
This repository has Python scripts to create a dataset for Paligemma to segment water in satellite images.

## Prerequisites 
- Python 3.11 
- Pipenv

## General set up and activation of the environment 
Clone this repository first.

Once the project is cloned, you need to create and set up a virtual environment. To do so,  
open a terminal and type the following commands:

```
> pipenv install --dev
> pipenv shell
> pre-commit install 
```

This last plugin will facilitate and automate the code formatting.

## Satellite images
The original dataset is in Kaggle. It can be downloaded from [this link](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies). 

Since some masks are not correct, I manually selected the correct images. You can read [this blog post]()
to see how I did it. You can also contact me to obtain the final set of images. 

***IMPORTANT: In the root directory, create a folder called data/ where you add the cleaned images and masks, each 
in its corresponding subfolder.***

## Create a dataset to fine-tune Paligemma

Once the cleaned images and masks are added in the dataset/ folder, you can run the following command 
to create the dataset used by Paligemma:

```
> python src/convert.py --data_path=<absolute path to data/> --masks_folder_name=<it must be a subfolder of data/> --images_folder_name=<it must be a subfolder of data/>
```
