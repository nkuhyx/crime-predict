import csv
import numpy as np
import xgboost as xgb

# save csv file with 8 decimal
def SaveCsv(csv_value, csv_head, csv_filename):
	with open(csv_filename, 'wb') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(csv_head)
		for i in range(csv_value.shape[0]):
			tmp = []
			tmp.append('{:.0f}'.format(i))
			for j in range(csv_value.shape[1]):
				tmp.append('{:.8f}'.format(csv_value[i][j]))
			writer.writerow(tmp)
	csvfile.close()
	print "save csv file complete~"

# xgboost model training
def XgModel(dtrain, save_name):
	# set paramaters
	param = {'bst:max_dept': 8, 'bst:eat': 0.1, 'silent': 1, 'objective': 'multi:softprob', 'num_class': 39, 'nthread': 4, 'eval_metric': 'mlogloss'}
	plst = param.items()
	num_round = 2000
	
	# training
	bst = xgb.train(plst, dtrain, num_round)
	print "train model complete~"
	bst.save_model(save_name)
	print "save model complete~"
	return bst

# xgboost predictin
def XgPredict(dtest, model, csv_head, csv_filename):
	ypred = model.predict(dtest)
	print "predict complete~"
	SaveCsv(ypred, csv_head, csv_filename)
	print "save result complete~"
	return 


