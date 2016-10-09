import csv
import numpy as np
import pandas as pd
import time
from datetime import datetime
#from sklearn import preprocessing

# categorys dict
def CateList():
	csvfile =  open('data/sampleSubmission.csv', 'r')
	categorys = csv.reader(csvfile, dialect = 'excel',delimiter = ',').next()
	csvfile.close()
	return categorys

# preciouse address of street
def TakeAddress(address, dim):
	if '/' in address:
		return address.strip().split('/')[dim].strip()
	else:
		return address.strip().split('of')[1].strip()
# street map
def StreetMap(St_list, x):
	if x not in St_list:
		St_list.append(x)
	return St_list.index(x)
		
# extract features from dates
def ParseTime(dtime):
	dt = np.zeros((6, len(dtime)))
	for i in range(len(dtime)):
		dd = datetime.strptime(dtime[i], "%Y-%m-%d %H:%M:%S")
		dt[0][i] = dd.year
		dt[1][i] = dd.month
		dt[2][i] = dd.day
		dt[3][i] = dd.hour
		dt[4][i] = abs(int(dd.minute) - 30)
		#dt[5][i] = time.mktime(time.strptime(dtime[i], "%Y-%m-%d %H:%M:%S"))
	return dt

# common operation for train and test
def ComFrame(filename, dim):
	cf = pd.read_csv(filename)

	# week dict
	week_dict = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,'Thursday': 4,'Friday': 5,'Saturday': 6,'Sunday': 7 }

	# week & address
	cf['DayOfWeek'] = cf['DayOfWeek'].apply(lambda x: week_dict[x])
	cf['Address'] = cf['Address'].apply(lambda x: int('/' in x))
	#cf['Address1'] = cf['Address']
	#cf['Address'] = cf['Address'].apply(lambda x: TakeAddress(x, 0))
	#cf['Address1'] = cf['Address1'].apply(lambda x: TakeAddress(x, 1))

	# xy to pca
	#xy_scaler = preprocessing.StandardScaler()
	#xy_scaler.fit(cf[['X', 'Y']])
	#cf[['X', 'Y']] = xy_scaler.transform(cf[['X', 'Y']])
	if dim:	
		cf = cf[cf['Y'] < 90]
		cf.index = range(len(cf))
	cf['X'] = cf['X'].apply(lambda x: x + 122)
	cf['Y'] = cf['Y'].apply(lambda x: x - 37)
	
	# time
	tfeature = ParseTime(cf['Dates'])
	cf = cf.drop(['Dates'], axis = 1)
	cf['Year'] = pd.Series(tfeature[0])
	cf['Month'] = pd.Series(tfeature[1])
	cf['Day'] = pd.Series(tfeature[2])
	cf['Time'] = pd.Series(tfeature[3])
	cf['Minute'] = pd.Series(tfeature[4])
	
	#cf['T'] = pd.Series(tfeature[5])
	#cf['T'] = (cf['T'] - cf['T'].mean())/(cf['T'].max() - cf['T'].min())
	return cf

def DataTrans(trainfile, testfile):
	categorys = CateList()[1:]
	# train data
	trains = ComFrame(trainfile, True)
	trains = trains.drop(['Descript'], axis = 1).drop(['Resolution'], axis = 1)
	trains['Category'] = trains['Category'].apply(lambda x: categorys.index(x))
	# PdDistrict
	Pd_list = list(trains['PdDistrict'].unique())
	trains['PdDistrict'] = trains['PdDistrict'].apply(lambda x: Pd_list.index(x))
	# Street

	# build Street list for mapping
	#St_list = list(trains['Address'].unique())
	#for x in list(trains['Address1'].unique()):
	#	if x not in St_list:
	#		St_list.append(x)

	# trains street mapping
	#trains['Address'] = trains['Address'].apply(lambda x: St_list.index(x))
	#trains['Address1'] = trains['Address1'].apply(lambda x: St_list.index(x))
	
	# test data
	tests = ComFrame(testfile, False)
	
	#for x in list(tests['Address'].unique()):
	#	if x not in St_list:
	#		St_list.append(x)

	tests['PdDistrict'] = tests['PdDistrict'].apply(lambda x: Pd_list.index(x))
	#tests['Address'] = tests['Address'].apply(lambda x: St_list.index(x))
	#tests['Address1'] = tests['Address']
	return [trains, tests.drop(['Id'], axis = 1), categorys]

		
