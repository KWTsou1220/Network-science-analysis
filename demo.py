from utils import read_train_data, read_test_data
from utils import extract_features, build_text_dict
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score

import networkx as nx
import numpy as np
import xgboost as xgb
import time
import math
import pickle
import pandas as pd

train_data_path = './Dataset/training_set.txt'
test_data_path = './Dataset/testing_set.txt'
node_info_path = './Dataset/node_information.csv'


# Loading data and feature extraction
G, node_pairs_train, t_train = read_train_data(train_data_path)
node_pairs_test = read_test_data(test_data_path)
node_info = build_text_dict(node_info_path)

print('Extract features from training data')
x_train = extract_features(G, node_pairs_train, node_info)

print('Extract features from test data')
x_test = extract_features(G, node_pairs_test, node_info)

#x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.4)

#np.save('./Dataset/x_train', x_train)
#np.save('./Dataset/x_test', x_test)


# Training
param = {'objective': 'multi:softmax', # Specify multiclass classification
         'num_class': 2, # Number of possible output classes
         'tree_method': 'gpu_hist' # Use GPU accelerated algorithm
        }  
start = time.time()
xgb_class = xgb.XGBClassifier(max_depth=25, n_estimators=1000, nthread=2, **param)
xgb_class.fit(x_train, t_train)
end = time.time()
print('Time: ', end-start)

#y_valid = xgb.predict(x_valid)
#print('F1 score: ', f1_score(y_pred=y_valid, y_true=t_valid, average='macro'))

# Saving model
'''
with open('Models/xgb.pickle', 'wb') as f:
    pickle.dump(xgb_class, f)
    

with open('save/clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    print(clf2.predict(X[0:1]))
'''

# Prediction
split_num = 3000
data_size = x_test.shape[0]
n_iter = math.ceil(data_size/split_num)
y_test = []
for idx in range(n_iter):
    y = xgb_class.predict(x_test[idx*split_num:(idx+1)*split_num])
    y_test += [y]

y_test = np.concatenate(y_test, axis=0)
data_size = y_test.shape[0]
id_ = np.arange(data_size)
df = pd.DataFrame(data=np.stack([id_, y_test], axis=1), columns=['id', 'category'])
df.to_csv('./Dataset/test_xgb_500_20.csv', index=False)