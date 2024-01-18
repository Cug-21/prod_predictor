from ai_model import load_market_data, preprocess_market_industry_data, feature_engineering_market_industry, load_transformer_model, feature_engineering
import requests
import re
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from openai import OpenAI
import os 
from dotenv import load_dotenv
import json

load_dotenv('key.env')

OpenAI.api_key=os.getenv('OPENAI_API_KEY')
client = OpenAI()


api_key = '357ec872a6f3ee1578464644e2f49e40'


def fetch_fred_data(series_id, api_key):
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return response.json()
    
unemployment_rate_data = fetch_fred_data('UNRATE', api_key)  # Unemployment Rate
inflation_rate_data = fetch_fred_data('T5YIFR', api_key)  # Inflation Rate (5-Year Forward Inflation Expectation Rate)
cpi_data = fetch_fred_data('CPIAUCSL', api_key)  # Consumer Price Index
ppi_data = fetch_fred_data('PPIACO', api_key)  # Producer Price Index
balance_of_trade_data = fetch_fred_data('BOPGSTB', api_key)  # Balance of Trade
bank_lending_data = fetch_fred_data('TOTLL', api_key)  # Bank Lending (Total Loans and Leases, All Commercial Banks)
interest_rate_data = fetch_fred_data('FEDFUNDS', api_key)  # Federal Funds Effective Rate
wage_growth_data = fetch_fred_data('CES0500000003', api_key)  # Average Hourly Earnings of All Employees, Total Private
retail_sales_data = fetch_fred_data('RSXFS', api_key)  # Retail and Food Services Sales
housing_market_data = fetch_fred_data('HOUST', api_key)  # Housing Start
government_debt_data = fetch_fred_data('GFDEBTN', api_key)  # Federal Government Debt: Total Public Debt
currency_exchange_data = fetch_fred_data('DEXUSEU', api_key)  # US Dollar to Euro Exchange Rate

bond_yield_data = fetch_fred_data('DGS10', api_key)  # 10-Year Treasury Bond Yield


economic_data = {
    'unemployment_rate': unemployment_rate_data,
    'inflation_rate_data': inflation_rate_data,
    'cpi_data': cpi_data,
    'interest_rate_data': interest_rate_data,
    'ppi_data': ppi_data,
    'bank_lending_data': bank_lending_data,
    'balance_of_trade_data': balance_of_trade_data,
    'wage_growth_data': wage_growth_data,
    'retail_sales_data': retail_sales_data,
    'housing_market_data': housing_market_data,
    'government_debt_data': government_debt_data,
    'currency_exchange': currency_exchange_data,

    'bond_yield': bond_yield_data
}

def preprocess_economic_data(raw_data):
    processed_data = {}
    
    for key, value in raw_data.items():
        if 'observations' in value:
            df = pd.DataFrame(value['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Filter data to include only entries from 2020 onwards
            df = df[df.index >= '2020-01-01']

            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df.dropna(inplace=True)

            processed_data[key] = df
        else:
            print(f"Error fetching data for {key}: {value.get('error_message', 'No error message available')}")

    return processed_data

def feature_engineering_economic(data):
    features = {}
    for key, df in data.items():
        annual_change = df['value'].pct_change(periods=12).reset_index().values.tolist()
        annual_change_dicts = [{"date": str(date), "value": value} for date, value in annual_change]
        features[f"{key}_annual_change"] = annual_change_dicts

    return features

processed_data = preprocess_economic_data(economic_data)

economic_features = feature_engineering_economic(processed_data)



def generate_chat_prompt(combined_data):
    summary_data = {}
    
    for key, data_list in combined_data.items():
        values = [item['value'] for item in data_list if 'value' in item]
        if values:
            series = pd.Series(values)
            summary_data[key] = {'mean': series.mean(), 'max': series.max(), 'min': series.min()}

    prompt = "Based on the following summarized economic data, rate the current economy's health on a scale of 0-10, where 0.0 is very poor and 10.0 is excellent, Only return the score:\n"
    prompt += json.dumps(summary_data)
    return prompt

def get_market_health_score(data, model="gpt-4"):
    prompt = generate_chat_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    print(response)
    if response.choices:
        response_text = response.choices[0].message.content.strip()
    else:
        response_text = "No response received."
   
    return response_text



def eco_score():

    processed_data = preprocess_economic_data(economic_data)
    economic_features = feature_engineering_economic(processed_data)

    score = get_market_health_score(economic_features)

    return score