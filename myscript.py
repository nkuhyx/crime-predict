import datas 
import methods 
import xgboost as xgb
import pandas as pd


# data section
trainfile = 'data/train.csv'
testfile = 'data/test.csv'
[trains, tests, categorys] = datas.DataTrans(trainfile, testfile)

dtrain = xgb.DMatrix(trains.drop(['Category'], axis = 1),\
label = trains['Category'])

dtest = xgb.DMatrix(tests)

# save data with xgb format
dtrain.save_binary('data/train.buffer')
dtest.save_binary('data/test.buffer')
print "data save complete~"

# load data 
#dtrain = xgb.DMatrix('data/train.buffer')
#dtest = xgb.DMatrix('data/test.buffer')

# method section
model_name = 'model/6_1000_05.model'
model = methods.XgModel(dtrain, model_name)
# load model
#model = xgb.Booster({'nthread': 4})
#model.load_model(model_name)
csv_filename = 'data/6_1000_05.csv'
categorys = datas.CateList() #
methods.XgPredict(dtest, model, categorys, csv_filename)

