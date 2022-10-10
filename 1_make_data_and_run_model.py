#### run before code:
### this updates the server:
## sudo apt update && sudo apt upgrade -y

import os

try:
    import botocore
    import boto3
except ImportError:
    print("No module named botocore or boto3. You may need to install boto3")
    sys.exit(1)
boto3.compat.filter_python_deprecation_warnings()
print("imported boto3 using compat.filter\nIt's still deprecated, so we still need to fix that.")

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("snowflake-connector-python")
install("matplotlib")
install("tensorflow")
install("sklearn")

import numpy as np
from datetime import timedelta
import datetime
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sklearn

# fix random seed for reproducibility
np.random.seed(7)

print("starting...")

# The Snowflake Connector library.
import snowflake.connector
import pandas as pd
import boto3
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

def get_last_run_date():
     ssm = boto3.client('ssm','us-west-2')
     response = ssm.get_parameter(
        Name="/rev-ml/revenue_model_date",
        WithDecryption=True
     ) 
     print(response['Parameter']['Value'])
     return response['Parameter']['Value']


credentials = get_credentials(ssm_params)
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
	for i in range(len(dataset)-look_back):
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


query1 = """SELECT DISTINCT
                DATE(orders.DELIVEREDON) as DATE,
                CASE WHEN split_part(billing_accounts.INVOICEEMAIL, '@', -1) IS NULL THEN 
                                LOWER(orders.CUSTOMEREMAILDOMAIN)
                    ELSE LOWER(split_part(billing_accounts.INVOICEEMAIL, '@', -1))
                    END as invoice_domain,
               -- CASE WHEN orders.ORDEREDUNDERCONTRACTID IS NULL THEN 0 ELSE 1 END as IS_CONTRACT,
                SUM(product_sales.FINALPRICE) as FINALPRICE

            FROM ANALYTICS_MAIN.DBO.PRODUCTSALES AS product_sales
            INNER JOIN ANALYTICS_MAIN.DBO.ORDERS AS orders 
                ON (product_sales."ORDERKEY") = (orders."ORDERKEY")
            LEFT JOIN ANALYTICS_MAIN.CUSTOMERINFO.DOMAININDUSTRYMAP AS domain_industry_classification 
                ON lower((orders."CUSTOMEREMAILDOMAIN")) = lower((domain_industry_classification."DOMAIN"))
            LEFT JOIN ANALYTICS_MAIN.DBO.BILLINGACCOUNTS AS billing_accounts 
                ON (orders."BILLINGACCOUNTID") = (billing_accounts."ID")
            WHERE
            -- (domain_industry_classification.INDUSTRY) IN ('Broadcasting', 'Media & Internet') AND 
            FINALPRICE > 0
            GROUP BY invoice_domain, DATE
            --, IS_CONTRACT
            ORDER BY invoice_domain, DATE
            ;
            """
print("starting to pull data from snowflake into the enviroment.")
cur = ctx.cursor().execute(query1)
df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])
print("data pulled from snowflake", df.shape)

df['FINALPRICE'] = df['FINALPRICE'].astype('float')
df['DATE'] = pd.to_datetime(df['DATE'])

##minimium observations needed to make a model
min_obs = 700

freq_domain = df['INVOICE_DOMAIN'].value_counts()
freq_domain = pd.DataFrame(freq_domain)
freq_domain.reset_index(inplace=True)
freq_domain = freq_domain.sort_values(by = 'INVOICE_DOMAIN', ascending=False)
freq_domain = freq_domain[(freq_domain['INVOICE_DOMAIN'] > min_obs)]
company_list = list(freq_domain['index'][(freq_domain['INVOICE_DOMAIN'] > min_obs)])

## these companies are public email domains
remove_companies = ['gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'me.com', 'comcast.net', 
                    'mac.com', 'msn.com', 'live.com', 'linkedin.com', 'sbcglobal.net', 'outlook.com', 
                    'icloud.com', 'earthlink.net', 'bill.com', 'verizon.net', 'att.net']

for i in range(0, len(remove_companies)):
    company_list.remove(remove_companies[i])


print("number of company domains that get a model:", len(company_list))

from datetime import timedelta
import datetime

## make a list of all possible windows using itertools
import itertools
from itertools import product

## this is the smallest date in the database
test_date = datetime.datetime.strptime('2010-09-01', "%Y-%m-%d")

END_list = pd.date_range(start=test_date, end=datetime.datetime.now(), freq='7D')
START_list = END_list - timedelta(days = 91)
test_date = str(test_date)

combination = pd.DataFrame(itertools.product(END_list, company_list))
combination = combination.rename(columns={0: "END",
                                         1: "company"})
## dataframe with start, end & company
combination['START'] = combination['END'] - timedelta(days = 91)
combination['START'] = pd.to_datetime(combination['START'])
combination['END'] = pd.to_datetime(combination['END'])
print("this is the final data length:", combination.shape)

## load all the prior data
try:
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket('revcom-sagemaker-prediction-outputs')
    prior_data = pd.DataFrame()

    for object_summary in my_bucket.objects.filter(Prefix="Revenue/moving_window"):
        if object_summary.key.endswith(".csv"):
            df3 = pd.read_csv("s3://revcom-sagemaker-prediction-outputs/{}".format(object_summary.key))
            prior_data = prior_data.append(df3, ignore_index=True)
#         print(object_summary.key, prior_data.shape, datetime.datetime.now())

    prior_data['START'] = pd.to_datetime(prior_data['START'])
    prior_data['END'] = pd.to_datetime(prior_data['END'])        
except:
    print("there is no prior data to load.", prior_data.shape, datetime.datetime.now())

prior_data = prior_data.drop_duplicates()

try:
    ## define windows not calculated - then send them into the loop
    combined = combination.merge(prior_data, how='left',
                                 left_on=['company', 'START', 'END'], 
                                 right_on = ['company', 'START', 'END'])
    loop_data = combined[combined['rolling_quarter'].isna() == True]
    loop_data = loop_data.reset_index(drop=True)
    loop_data = loop_data.sort_values(by=['company', 'START'])
    print("loop_data", loop_data.shape, "prior_data", prior_data.shape, "combination", combination.shape)
    
except: ## if there is no data to load, run all the loops
    print("there was no prior data to exclude from the loop")
    loop_data = combination
    
## this does a quarter moving window that 'final price' is calculated every 7 days
df2 = pd.DataFrame([])

if loop_data.shape[0] > 10:
    for ii in range(0, len(loop_data['company'])):
        company = loop_data['company'][ii]
        START = loop_data['START'][ii]
        END = loop_data['END'][ii]

        FINAL = df['FINALPRICE'][(df['DATE'] < END) & (df['DATE'] > START) &
                                         (df['INVOICE_DOMAIN'] == company)].sum()
        sent1 = pd.DataFrame(data = {'company': [company],
                                     'rolling_quarter': [FINAL],
                                     'START': [START],
                                     'END': [END]})
        df2 = df2.append(sent1)

        if ii % 100 == 0: ## prints out progress
            print("building the moving window, ", round(ii / len(loop_data['company']), 2)*100, "% done.", ii, len(loop_data['company']), datetime.datetime.now())
        if ii % 500 == 0: ## saves
            file_name = '{date}.csv'.format(date = str(datetime.datetime.today())[0:19].replace(" ", "_"))
            # save out the file to S3 bucket for matillion pick-up
            df2.to_csv('s3://revcom-sagemaker-prediction-outputs/Revenue/moving_window/{}'.format(file_name),
                      index = False)
            df2 = pd.DataFrame([])
            print("file saved out", datetime.datetime.now())
        if ii == len(loop_data['company'])-1: ## saves
            file_name = '{date}.csv'.format(date = str(datetime.datetime.today())[0:19].replace(" ", "_"))
            # save out the file to S3 bucket for matillion pick-up
            df2.to_csv('s3://revcom-sagemaker-prediction-outputs/Revenue/moving_window/{}'.format(file_name),
                      index = False)
            df2 = pd.DataFrame([])
            print("file saved out", datetime.datetime.now())


## dont want any moving windows that end prior to the current date
## load all the prior data
if loop_data.shape[0] > 0:
    ### we need to load all the data into a dataframe
    try:
        s3 = boto3.resource('s3')
        my_bucket = s3.Bucket('revcom-sagemaker-prediction-outputs')
        df2 = pd.DataFrame()

        for object_summary in my_bucket.objects.filter(Prefix="Revenue/moving_window"):
            if object_summary.key.endswith(".csv"):
                df3 = pd.read_csv("s3://revcom-sagemaker-prediction-outputs/{}".format(object_summary.key))
                df2 = df2.append(df3, ignore_index=True)
#             print(object_summary.key, df2.shape)       
    except:
        print("there is no data to load, something is really broken.", df2.shape)

if loop_data.shape[0] == 0:   
    ## the data is already loaded into a dataframe and it needs to be renamed
    df2 = prior_data
    print("There was no new data added", df2.shape)
    
df2 = df2.drop_duplicates()
df2['START'] = pd.to_datetime(df2['START'], infer_datetime_format=True)
df2['END'] = pd.to_datetime(df2['END'], infer_datetime_format=True)

## notice how its joined by company & start / end so its a quarter over quarter change
out1 = df2.merge(df2, left_on=['company', 'START'], right_on=['company', 'END'])

out1 = out1.rename(columns={"rolling_quarter_x":"rolling_quarter",
                            "START_x":"START",
                            "END_x":"END",
                            "rolling_quarter_y":"rolling_quarter_historic",
                            "START_y":"START_historic",
                            "END_y":"END_historic"})

## historic quarter over quarter percent change
out1['percent_1'] = (out1['rolling_quarter'] - out1['rolling_quarter_historic']) / (out1['rolling_quarter'] + out1['rolling_quarter_historic'])

## columns: START 	END 	FINALPRICE 	company 	FINALPRICE_lag 	percent_1
out1 = out1.rename(columns={"rolling_quarter":"FINALPRICE",
                        "rolling_quarter_historic":"FINALPRICE_lag"})
print("done with the historic data making", out1.shape)

######################################################################
### the model code:
df1 = out1[['START', 'END', 'FINALPRICE', 'company', 'FINALPRICE_lag', 'percent_1']].drop_duplicates()
company_list = df2['company'].unique()


#date_time = '09_14_2022' ## version of the model saved out
date_time = get_last_run_date()
file_name2 = 'RNN_grid_search_{DATE}_.csv'.format(DATE = date_time)

try:
    a = pd.read_csv('s3://revcom-sagemaker-prediction-outputs/Revenue/model_parameters/'+file_name2)
except:
    sys.exit('you dont have a trained model to do that. pick a different date.')
    
out_1 = a[a['train_RMSE'].isna() == False].sort_values(by = ['company_list', 'test_RMSE'], ascending=True)
out_1 = out_1.groupby('company_list').first().reset_index()
print("how many models are we runing and how many paramters did we check?", out_1.shape)

## need to save the perdictions somewhere
df1['predicted_revenue'] = 0
directory = 's3://revcom-sagemaker-prediction-outputs/Revenue/model_parameters/RNN_models'+date_time+'/'
print("the models live here - right?", directory)

for ii in range(0, len(out_1['company_list'])):

    company = out_1['company_list'][ii]
    look = out_1['look_back'][ii]
    epoch = out_1['epochs'][ii]
    batch = out_1['batch_size'][ii]

    # load the dataset
    dataset = df1['FINALPRICE'][df1['company'] == company].values
#     print(company, "has this much data:", len(df1['FINALPRICE'][df1['company'] == company]))
    dataset = dataset.astype('float32')
    
    if len(df1['FINALPRICE'][df1['company'] == company]) <= 10:
        print("WARNING: there is only {obs} observations.  And {company} will not get a prediction vector.".format(obs = len(dataset), company = company))

    if len(df1['FINALPRICE'][df1['company'] == company]) > 10:
        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset.reshape(-1, 1))

        # dont need to split into train and test sets because this is prod
        # reshape into X=t and Y=t+1
        trainX = create_prod_dataset(dataset = dataset, look_back = look)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        ###################################
        # load the LSTM network
        # load model from JSON

        s3 = boto3.client('s3')
        Bucket= 'revcom-sagemaker-prediction-outputs'
        Key='Revenue/model_parameters/RNN_models'+date_time+'/' ## needs trailing /
        Filename = str(company).split('.')[0] + "_lookback" +str(look) + "_epoch" + str(epoch) + "_batch" + str(batch) 

        s3.download_file(Bucket, 
                         Key+Filename+'.json',
                         Filename+'.json')
        ## The "filename" part is the output filename.
        ## The Key part requires the full key, including the filename

        ## load the structure of the RNN via json file
        json_file = open(Filename+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json.replace('\\', ''). strip('"')) 

        ## load the weights from HDF5
        s3.download_file(Bucket, Key+Filename+'.h5', Filename+'.h5') #Works!
        loaded_model.load_weights(Filename+'.h5')

        ###############################################
        # make predictions
        trainPredict = loaded_model.predict(trainX, verbose=0)

        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)

        ## add an extra row for the weekly perdiction into the future
        df_temp = {'company': company, 
               'FINALPRICE': np.nan, 
               'FINALPRICE_lag': np.nan,
               'percent_1': np.nan,
               'START': max(df1.START[df1['company'] == company])+pd.DateOffset(days=7),
               'END': max(df1.END[df1['company'] == company])+timedelta(days=7),
               'predicted_revenue': np.nan}
        df1 = df1.append(df_temp, ignore_index = True)

        ## Pad the perdiction with 0's for the lookback time 
        ss0ss = pd.Series([0]).repeat(look)
        perdiction_vector = pd.concat([ss0ss, pd.DataFrame(trainPredict)])

        ## put the prediction into the dataframe for safe keeping

        df1.loc[(df1['company'] == company), 'predicted_revenue'] = list(perdiction_vector[0])

        print(round(ii / len(out_1['company_list']), 2), company, look, epoch, batch, datetime.datetime.now())
        del loaded_model
        os.remove(Filename+'.h5')
        os.remove(Filename+'.json')

df1 = df1.sort_values(by=['company','START'])
df1 = df1.reset_index(drop=True)
print("the final data output dims, should be bigger each week:", df1.shape)

file_name = '{date}.csv'.format(date = str(datetime.datetime.today())[0:19].replace(" ", "_"))
#save out the file to S3 bucket for matillion pick-up
df1[['START', 'END', 'FINALPRICE', 'company', 'FINALPRICE_lag', 'percent_1', 'predicted_revenue']].to_csv(
    's3://revcom-sagemaker-prediction-outputs/Revenue/historic/alarms/{}'.format(file_name), 
    index=False)