# Car Price Prediction

## Changes

- removed some bugs in app.py
- trained the model once and saved using joblib, now we are loading the already trained model in every time the website/server starts
- made the 'Make' user-input with 'dropdown' instead of 'input'

## TODO

- the model performance is not good(empiricaly)
- use one-hot-encoding instead of label-encoding for nominal data(here 'Make', 'Owner', 'FuelType')
