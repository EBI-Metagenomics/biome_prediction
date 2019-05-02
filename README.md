# GOLD biome predictor
This project contains a library and command line tool to predict a GOLD biome for a given set of text.

## Installation
```bash
git clone git@github.com:EBI-Metagenomics/biome_prediction.git
pip install .
```
## Usage (i.e: command)
```bash
predict-biome
```
This will open a prompt into which text can be typed/pasted for prediction.

## Updating training data:
```bash
bash fetch_data/update_data.sh
```
This will create a new set of training data; move it replace raw_data.tsv in the data/ folder, 
and re-run classify.py to retrain the model. Remember to commit the updated model to GitHub!