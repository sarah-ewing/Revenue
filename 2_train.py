## training the model for each domain.
### this is so much fun .... said no one ever.

## the boil

## Refactored and reordered imports to reduce overhead.  And confusion.
import os
import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("snowflake-connector-python")
install("matplotlib")
install("tensorflow")
install("sklearn")
install("s3fs")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "s3fs"])
import json
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import snowflake.connector
import boto3
from s3fs import S3FileSystem
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sklearn
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
boto3.compat.filter_python_deprecation_warnings()
print("imported boto3 using compat.filter\nIt's still deprecated, so we still need to fix that.")

# fix random seed for reproducibility
np.random.seed(7)

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("snowflake-connector-python")

print("starting...")

# The Snowflake Connector library.
pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_colwidth', None)

# Get the credentials for Snowflake
ssm_params = ['/rev-ml/sagemaker/snowflake-user', '/rev-ml/sagemaker/snowflake-password']
def get_credentials(params):
    ssm = boto3.client('ssm','us-west-2')
    response = ssm.get_parameters(
        Names=params,
        WithDecryption=True
    ) 
    #Build dict of credentials
    param_values={k['Name']:k['Value'] for k in  response['Parameters']}
    return param_values

credentials = get_credentials(ssm_params)

# gets the last run datetime from aws parameter store
def get_last_run_date():
     ssm = boto3.client('ssm','us-west-2')
     response = ssm.get_parameter(
        Name="/rev-ml/revenue_model_date",
        WithDecryption=True
     ) 
     print(response['Parameter']['Value'])
     return response['Parameter']['Value']

def set_last_run_date(date_string):
     ssm = boto3.client('ssm','us-west-2')
     ssm.put_parameter(
        Name="/rev-ml/revenue_model_date",
        Value=date_string,
        Overwrite=True
     )
     return

# Gets the version
ctx = snowflake.connector.connect(
                                user=credentials['/rev-ml/sagemaker/snowflake-user'],
                                password=credentials['/rev-ml/sagemaker/snowflake-password'],
                                account='msa72542'
                                )
ctx.cursor().execute('USE warehouse DATAQUERY_WH')
print("done.")

## functions needed to run:
# convert an array of values into a dataset matrix
def create_training_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def create_prod_dataset(dataset, look_back):
	dataX = []
	for i in range(len(dataset)-look_back+1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
	return np.array(dataX)

##################################
## the training
print("load the training data")
try:
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket('revcom-sagemaker-prediction-outputs')
    df2 = pd.DataFrame()

    for object_summary in my_bucket.objects.filter(Prefix="Revenue/moving_window"):
        if object_summary.key.endswith(".csv"):
            df3 = pd.read_csv("s3://revcom-sagemaker-prediction-outputs/{}".format(object_summary.key))
            df2 = df2.append(df3, ignore_index=True)
#         print(object_summary.key, df2.shape)       
except:
    print("there is no data to load, something is really broken.", df2.shape)

df2 = df2.drop_duplicates()
df2['START'] = pd.to_datetime(df2['START'], infer_datetime_format=True)
df2['END'] = pd.to_datetime(df2['END'], infer_datetime_format=True)

out1 = df2.drop_duplicates()
company_list = out1['company'].unique()
df2 = df2.rename(columns={"rolling_quarter":"FINALPRICE"})
df2 = df2[['START', 'END', 'FINALPRICE', 'company']].drop_duplicates()
print("the data shape", df2.shape)

#########################################################################
print("load the grid search for the trained models")

date_time = get_last_run_date()
file_name2 = 'RNN_grid_search_{DATE}_.csv'.format(DATE = date_time)
directory = 's3://revcom-sagemaker-prediction-outputs/Revenue/model_parameters/RNN_models'+date_time+'/'
print("old file: ", file_name2)
print("data directory: ", directory)

s3 = boto3.client('s3')
bucket_name = "revcom-sagemaker-prediction-outputs"
folder_name = "Revenue/model_parameters/RNN_models"+date_time
s3.put_object(Bucket=bucket_name, Key=(folder_name+'/'))
  
try:
    a = pd.read_csv('s3://revcom-sagemaker-prediction-outputs/Revenue/model_parameters/'+file_name2)
    empty_cell = len(a.train_RMSE[(a['train_RMSE'].isna() == True)])
    print("loaded prior training", a.shape, "and it is {}% done".format(round(empty_cell/a.shape[0], 2)))

except:
    print("WARNING: no prior training.")
#     import itertools

#     company_list
#     look_back = [5, 15, 25] ## weeks
#     epochs=[50, 100]#, 100, 150, 200, 500]
#     batch_size=[10, 30, 50, 100]

#     a = [company_list, look_back, epochs, batch_size]
#     a = pd.DataFrame(itertools.product(*a))
#     a['train_RMSE'] = None
#     a['test_RMSE'] = None
#     a = a.rename(columns={0: "company_list", 1: "look_back", 2: "epochs", 3: "batch_size"})
#     a.head()
#     print(a.shape)
#     a.to_csv('s3://revcom-sagemaker-prediction-outputs/Revenue/model_parameters/'+file_name2, 
#                  index=False)
 
STARRT = min(a.index[(a['train_RMSE'].isna() == True)], default=0)
ENDDD = len(a['company_list'])
print("start and end", STARRT, ENDDD)
print("################################# THE TRAINING #################################")

# Set the new date and store in AWS parameterStore
now = datetime.now() # current date and time
date_time = now.strftime("%m_%d_%Y")
set_last_run_date(date_time)
file_name2 = 'RNN_grid_search_{DATE}_.csv'.format(DATE = date_time)
print("start building new model file ",file_name2)
for ii in range(STARRT, ENDDD):

    company = a['company_list'][ii]
    look = a['look_back'][ii]
    epoch = a['epochs'][ii]
    batch = a['batch_size'][ii]

    # load the dataset
    dataset = df2['FINALPRICE'][df2['company'] == company].values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1
    look_back = look
    trainX, trainY = create_training_dataset(dataset = train, look_back = look_back)
    testX, testY = create_training_dataset(dataset = test, look_back = look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs = epoch, batch_size = batch, verbose=0)

    file_name = str(company).split('.')[0] + "_lookback" +str(look) +\
                "_epoch" + str(epoch) + "_batch" + str(batch) 

    # serialize model to JSON
    model_json = model.to_json()
    path_to_s3_object = directory+file_name

    s3 = S3FileSystem()
    with s3.open(path_to_s3_object+".json", 'w') as file:
        json.dump(model_json, file)

    # serialize weights to HDF5
    model.save(file_name+'.h5')  # creates a HDF5 file
    client = boto3.client('s3')
    client.upload_file(Filename=file_name+'.h5',
                      Bucket='revcom-sagemaker-prediction-outputs',
                      Key='Revenue/model_parameters/RNN_models'+date_time+'/'+file_name+'.h5')

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

    print(ii, round(ii / len(a['company_list']), 2), company, look, epoch, batch)
    a.loc[ii, ('train_RMSE')] = trainScore
    a.loc[ii, ('test_RMSE')] = testScore
    
    del model
    os.remove(file_name+'.h5') 

    a.to_csv('s3://revcom-sagemaker-prediction-outputs/Revenue/model_parameters/'+file_name2, 
             index=False)