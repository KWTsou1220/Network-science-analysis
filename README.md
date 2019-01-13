# Citation Network Link Prediction

We try to use the xgboost to predict the missing links.
This directory contains following files:

1. demo.py: main function
2. demo.ipynb: main function (jupyter file)
3. utils.py: various functions including
    - read_train_data: loadinig training data
    - read_test_data: loading test data
    - extract_features: feature extraction
    - other functions to get network features such as common_neighbor and jaccard_coeff