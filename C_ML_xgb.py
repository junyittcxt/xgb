import pandas as pd
#import tensorflow as tf
#from tensorflow import keras
import os
#import keras.backend as K
#from tensorflow.keras.optimizers import SGD
from sklearn.utils import class_weight
from sklearn import preprocessing
from sklearn.externals import joblib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import functools
#from tensorflow.python.client import device_lib
import sys, io
import time
import shutil
import xgboost as xgb
from xgboost import XGBClassifier
import ml_functions as ml

##########################
# PARAMETER
##########################  
asset = "TLT"
suffix = "_R_5"
input_file_name = "MLData_" + asset + suffix + ".csv"
target_col = "f_" + asset + "_1"

class_a_threshold = 0.001
split1 = 0.7
split2 = 0.5
input_file = "_data/" + input_file_name
folder_name = "XGB_" + target_col
  
##########################
# Log Initialization
##########################  
t0 = time.time()

##########################
# Data Preparation
##########################
df = pd.read_csv(input_file, index_col= "Date")
target_col_numeric = df[target_col]
target_col_bool =  ml.return_to_class(df[target_col], threshold = class_a_threshold) #Convert to binary class
df[target_col] = target_col_bool

##########################
### CHECK NA 
##########################
check_na = df.isnull().values.any()

if check_na:
    assert False, "Missing value(s) in training data!"

print("========DATA PARTITION & SCALING===========")
all_dates = df.index.values
y_array = df[[target_col]].values[:,0]
X_array = df.drop([target_col], axis = 1).values

#Use lesser columns
#X_df = df.drop([target_col], axis = 1)
#filter_col_1 = [col for col in X_df if col.startswith('p_TLT')]
##filter_col_2 = [col for col in X_df if col.startswith('p_esp')]
#filter_col = filter_col_1 #+ filter_col_2
#sub_X_df = X_df[filter_col]
#X_array = sub_X_df.values

x_train_r, x_test, y_train, y_test = ml.split(X_array, y_array, split1)
x_val_r, x_test_r, y_val, y_test = ml.split(x_test, y_test, split2)

#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#x_train = min_max_scaler.fit_transform(x_train_r)
#x_val = min_max_scaler.transform(x_val_r)
#x_test = min_max_scaler.transform(x_test_r)
#x_full = min_max_scaler.transform(X_array)

x_train = x_train_r
x_val = x_val_r
x_test = x_test_r
x_full = X_array

dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_val, label=y_val)
dtest = xgb.DMatrix(x_test, label=y_test)
dfull = xgb.DMatrix(X_array, label=y_array)

print("Training shape: ", x_train.shape, " ", y_train.shape)
print("Validation shape: ", x_val.shape, " ", y_val.shape)
print("Testing shape: ", x_test.shape, " ", y_test.shape)
print("Date Splits: {}, {}, {}, {}".format(all_dates[0], all_dates[x_train.shape[0]], all_dates[x_train.shape[0]+ x_val.shape[0]], all_dates[-1]))
print("================================\n\n")


print("========CLASS WEIGHTS===========")
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
class_weights = class_weights/class_weights[0]
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)
print("================================\n\n")

print("======== TRAINING ===============")
param = {
        'booster': 'dart',
         'eta': 0.05, 'gamma':2, 'max_depth': 3, 
         'min_child_weight': 1, 'max_delta_step':1,
         'silent': 1, 
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'lambda': 1,
         'alpha': 0,         
         'scale_pos_weight': class_weights_dict[1]*1.0,
#         'eval_metric': 'auc',
         'eval_metric': 'error',
#         'eval_metric': 'logloss',
         'objective': 'binary:logistic',
         'nthread': 4,
         }
num_round = 100
evallist = [(dtrain, 'train'), (dval, 'eval')]
evals_result = {}
model = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds = 10, evals_result = evals_result)

xgb.plot_importance(model, max_num_features=20, height=0.8)
plt.show()

stop_round = np.argmax(evals_result["eval"][param["eval_metric"]]) + 2
if param["eval_metric"] == "error":    
    stop_round = np.argmin(evals_result["eval"][param["eval_metric"]]) + 2
model = xgb.train(param, dtrain, stop_round, evallist, evals_result = evals_result)



#model = XGBClassifier()
#model.load_model('0001.model')  # load data
#print(evals_result['eval']['logloss'])
threshold = 0.50
print("================================\n\n")

print("======== PERFORMANCE ===============")
print(model.eval(dval))
print(model.eval(dtrain))

print("\n==============#TRAIN PERFORMANCE===========")
train_cm = ml.performance_binary(y_train, model.predict(dtrain), threshold)
print("\n==============#VALIDATION PERFORMANCE===========")
val_cm = ml.performance_binary(y_val, model.predict(dval), threshold)
print("\n==============#TEST PERFORMANCE===========")
test_cm = ml.performance_binary(y_test, model.predict(dtest), threshold)
print("================================\n\n")

    
print("======== SAVING INITIALIZATION: ===============")
z = 1
output_folder = folder_name + "_" + str(z).zfill(2)
while os.path.isdir(output_folder):
    output_folder = folder_name + "_" + str(z).zfill(2)
    z = z+1
    
try: 
    os.mkdir(output_folder)
except: 
    print("\n") 
    
print("======== SAVING PERFORMANCE SUMMARY ======")
cm_df = pd.DataFrame([train_cm, val_cm, test_cm])
cm_df["sample"] = ["train", "val", "test"]
cm_df.set_index(["sample"]).to_csv(os.path.join(output_folder, "a1_classification_performance.txt"))

print("======== SAVING PLOT ===============")
ml.threshold_sensitivity_plot(model, y_val, dval, output_folder, sample = "Validation")
ml.threshold_sensitivity_plot(model, y_test, dtest, output_folder, sample = "Test")

plt.scatter(x = model.predict(dval), y = y_val)
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Validation: Actual vs Predicted')
plt.grid(True)
plt.savefig(os.path.join(output_folder,'a2_Validation_Actual_vs_Predicted.png'))
plt.clf()

plt.scatter(x = model.predict(dtest), y = y_test)
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Testing: Actual vs Predicted')
plt.grid(True)
plt.savefig(os.path.join(output_folder,'a2_Test_Actual_vs_Predicted.png'))
plt.clf()
print("================================\n\n")

print("======== SAVING MODEL ===============")   
model.save_model(os.path.join(output_folder, 'm1_model.model'))
#model.dump_model(os.path.join(output_folder, 'm2_dump_model.txt'))

print("======== SAVE BACKTEST SIGNAL ===============")
pred_df = df[[target_col]]
pred_df["raw_signal"] = np.array(model.predict(dfull)).flatten()
pred_df["signal"] = 1*(np.array(model.predict(dfull)).flatten() >= 0.5)
pred_df.to_csv(os.path.join(output_folder, "signal.csv"))
print("================================\n\n")

print("======== SAVE PYTHON SCRIPT and INPUT DATA for reproducibility===============")
#Copy Python Script
shutil.copy2("C_ML_xgb.py", os.path.join(output_folder, "c1_C_ML_xgb.py"))
#Copy Data File
shutil.copy2(input_file, os.path.join(output_folder, input_file_name))
print("================================\n\n")

print("======== SAVE PYTHON SCRIPT and INPUT DATA for reproducibility===============")

ep50 = ml.performance_binary(y_test, model.predict(dtest), 0.5, silence = True)["excess_precision"]
ep50 = str(np.around(ep50*100, 2))
    
ep52 = ml.performance_binary(y_test, model.predict(dtest), 0.52, silence = True)["excess_precision"]
ep52 = str(np.around(ep52*100, 2))

ep55 = ml.performance_binary(y_test, model.predict(dtest), 0.55, silence = True)["excess_precision"]
ep55 = str(np.around(ep55*100, 2))

ep50v = ml.performance_binary(y_val, model.predict(dval), 0.5, silence = True)["excess_precision"]
ep50v = str(np.around(ep50v*100, 2))
    
ep52v = ml.performance_binary(y_val, model.predict(dval), 0.52, silence = True)["excess_precision"]
ep52v = str(np.around(ep52v*100, 2))

ep55v = ml.performance_binary(y_val, model.predict(dval), 0.55, silence = True)["excess_precision"]
ep55v = str(np.around(ep55v*100, 2))

val_ep = "XGB_EP_val_" + ep50v + "_" + ep52v + "_" + ep55v
test_ep = "XGB_EP_test_" + ep50 + "_" + ep52 + "_" + ep55

with open(os.path.join(output_folder, val_ep), 'w') as outfile: 
    print("")
with open(os.path.join(output_folder, test_ep), 'w') as outfile: 
    print("") 
        
dest_folder_name = "XGB_EP_" + ep50 + "_" + ep52 + "_" + ep55 + "_" + output_folder

if(os.path.isdir(dest_folder_name)):
    shutil.rmtree(output_folder)
    print("Destination path already exists!")
else:
    shutil.move(output_folder, dest_folder_name)
    
print(test_ep)
