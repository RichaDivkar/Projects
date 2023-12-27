# Code for ETL operations on Country-GDP data

# Importing the required libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime 

url = 'https://en.wikipedia.org/wiki/List_of_largest_banks'
# Exchange_rate_url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-PY0221EN-Coursera/labs/v2/exchange_rate.csv'
table_attribs = ["Name", "MC_USD_Billion"]
db_name = 'Banks.db'
table_name = 'Largest_banks'
output_path = './Largest_banks_data.csv'

# EXG_table = pd.read_csv(Exchange_rate_url,index_col = False)
# print(EXG_table)


def log_progress(message):
    ''' This function logs the mentioned message of a given stage of the
    code execution to a log file. Function returns nothing'''
    timestamp_format = '%Y-%h-%d-%H:%M:%S' 
    # Year-Monthname-Day-Hour-Minute-Second 
    now = datetime.now() # get current timestamp 
    timestamp = now.strftime(timestamp_format) 
    with open("./code_log.txt","a") as f: 
        f.write(timestamp + ' : ' + message + '\n')

def extract(url, table_attribs):
    ''' This function aims to extract the required
    information from the website and save it to a data frame. The
    function returns the data frame for further processing. '''

    page = requests.get(url).text
    data = BeautifulSoup(page,'html.parser')
    df = pd.DataFrame(columns=table_attribs)
    tables = data.find_all('tbody')
    rows = tables[0].find_all('tr')
    for row in rows:
        col = row.find_all('td')
        if len(col)!=0:
            if col[1].find('a') is not None:
                name = col[1].text.strip()
                mc_usd_billion_text = col[2].contents[0].replace('\n', '').replace(',', '').strip()
                
                if mc_usd_billion_text:
                    mc_usd_billion = float(mc_usd_billion_text)
                    data_dict = {"Name": name, "MC_USD_Billion": mc_usd_billion}
                    df1 = pd.DataFrame(data_dict, index=[0])
                df = pd.concat([df,df1], ignore_index=True)
    return df

def transform(df, csv_path):
    ''' This function accesses the CSV file for exchange rate
    information, and adds three columns to the data frame, each
    containing the transformed version of Market Cap column to
    respective currencies'''
    Exchange_rate_df = pd.read_csv(csv_path)
    
    df['MC_GBP_Billion'] = [np.round(x*Exchange_rate_df.loc[Exchange_rate_df['Currency'] == 'GBP', 'Rate'].values[0],2) 
                            for x in df['MC_USD_Billion']]
    df['MC_EUR_Billion'] = [np.round(x*Exchange_rate_df.loc[Exchange_rate_df['Currency'] == 'EUR', 'Rate'].values[0],2) 
                            for x in df['MC_USD_Billion']]
    df['MC_INR_Billion'] = [np.round(x*Exchange_rate_df.loc[Exchange_rate_df['Currency'] == 'INR', 'Rate'].values[0],2) 
                            for x in df['MC_USD_Billion']]

    return df

def load_to_csv(df, output_path):
    ''' This function saves the final data frame as a CSV file in
    the provided path. Function returns nothing.'''
    df.to_csv(output_path)

def load_to_db(df, sql_connection, table_name):
    ''' This function saves the final data frame to a database
    table with the provided name. Function returns nothing.'''
    df.to_sql(table_name, sql_connection, if_exists='replace', index=False)

def run_query(query_statement, sql_connection):
    ''' This function runs the query on the database table and
    prints the output on the terminal. Function returns nothing. '''
    print(query_statement)
    query_output = pd.read_sql(query_statement, sql_connection)
    print(query_output)

''' Here, you define the required entities and call the relevant
functions in the correct order to complete the project. Note that this
portion is not inside any function.'''
log_progress('Preliminaries complete. Initiating ETL process')

df = extract(url, table_attribs)

log_progress('Data extraction complete. Initiating Transformation process')

df = transform(df,'./exchange_rate.csv')

log_progress('Data transformation complete. Initiating loading process')

load_to_csv(df,output_path)

log_progress('Data saved to CSV file')

sql_connection = sqlite3.connect(db_name)

log_progress('SQL Connection initiated.')

load_to_db(df, sql_connection, table_name)

log_progress('Data loaded to Database as table. Running the query')

query_statement1 = f"SELECT * from {table_name}"
run_query(query_statement1, sql_connection)

query_statement2 = f"SELECT AVG(MC_GBP_Billion) from {table_name}"
run_query(query_statement2, sql_connection)


query_statement3 = f"SELECT Name from {table_name} LIMIT 5"
run_query(query_statement3, sql_connection)

log_progress('Process Complete.')

sql_connection.close()