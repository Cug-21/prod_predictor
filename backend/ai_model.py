import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
import re
import requests
import torch
import joblib
import yfinance as yf




def load_financial_data(ticker):
    stock = yf.Ticker(ticker)
    historical_data = stock.history(period="max")
    income_stmt = stock.income_stmt
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow
    earnings = stock.earnings_dates
    quarterlyfin = stock.quarterly_financials
    return {
        "historical_data": historical_data,
        "income_stmt": income_stmt,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
        "earnings": earnings,
        "quarterlyfin": quarterlyfin
    }


def create_financial_summary(data):
    historical_summary = summarize_historical_data(data["historical_data"])
    income_summary = summarize_income_statement(data["income_stmt"])
    balance_sheet_summary = summarize_balance_sheet(data["balance_sheet"])
    cashflow_summary = summarize_cashflow(data["cashflow"])
    earning_summary = summarize_cashflow(data["earnings"])
    quarterlyfin_summary = summarize_quarterlyfin(data["quarterlyfin"])
    print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
    print(quarterlyfin_summary)
    print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
    financial_summary = {
        "historical_summary": historical_summary,
        "income_summary": income_summary,
        "balance_sheet_summary": balance_sheet_summary,
        "cashflow_summary": cashflow_summary,
        "earning_summary": earning_summary,
        "quarterlyfin_summary": quarterlyfin_summary,

    }
    financial_summary['Close'] = data["historical_data"]['Close']
    return financial_summary

def summarize_historical_data(historical_data):
    summary = {}

    # Average closing price calculations
    summary['avg_close_30d'] = historical_data['Close'].tail(30).mean()
    summary['avg_close_90d'] = historical_data['Close'].tail(90).mean()
    summary['avg_close_1yr'] = historical_data['Close'].tail(252).mean()  # Approx. trading days in a year

    # High and low closing price
    summary['max_close'] = historical_data['Close'].max()
    summary['min_close'] = historical_data['Close'].min()

    # Trading volume stats
    summary['avg_volume_30d'] = historical_data['Volume'].tail(30).mean()
    summary['avg_volume_90d'] = historical_data['Volume'].tail(90).mean()

    # Price change
    summary['price_change_1d'] = historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[-2]
    summary['price_change_30d'] = historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[-30]

    # Volatility (standard deviation of closing prices)
    summary['volatility_30d'] = historical_data['Close'].tail(30).std()
    summary['volatility_90d'] = historical_data['Close'].tail(90).std()

    return summary

def summarize_income_statement(income_stmt):
    summary = {}
    income_stmt = income_stmt.T

    summary['total_revenue'] = income_stmt.get('Total Revenue', income_stmt.get('Revenue'))
    summary['gross_profit'] = income_stmt.get('Gross Profit')
    summary['operating_income'] = income_stmt.get('Operating Income')
    summary['net_income'] = income_stmt.get('Net Income')

    summary['ebit'] = income_stmt.get('EBIT')
    summary['ebitda'] = income_stmt.get('EBITDA')
    summary['interest_expense'] = income_stmt.get('Interest Expense')
    summary['net_interest_income'] = income_stmt.get('Net Interest Income')
  
    if summary['total_revenue'] is not None and summary['net_income'] is not None:
        summary['net_profit_margin'] = summary['net_income'] / summary['total_revenue']

    summary['research_development'] = income_stmt.get('Research Development')
    summary['selling_general_administrative'] = income_stmt.get('Selling General and Administrative')
    summary['operating_expense'] = income_stmt.get('Total Operating Expenses')

    summary['diluted_eps'] = income_stmt.get('Diluted EPS')
    summary['basic_eps'] = income_stmt.get('Basic EPS')

    summary['tax_provision'] = income_stmt.get('Tax Provision')
    summary['pretax_income'] = income_stmt.get('Pretax Income')
    summary['cost_of_revenue'] = income_stmt.get('Cost Of Revenue')

    return summary



def summarize_balance_sheet(balance_sheet):
    summary = {}
    balance_sheet = balance_sheet.T

    summary['total_assets'] = balance_sheet.get('Total Assets')
    summary['total_liabilities'] = balance_sheet.get('Total Liab', balance_sheet.get('Total Liabilities'))
    summary['total_shareholder_equity'] = balance_sheet.get('Total Stockholder Equity', balance_sheet.get('Total Equity'))
    
    # Current assets and liabilities
    summary['current_assets'] = balance_sheet.get('Total Current Assets')
    summary['current_liabilities'] = balance_sheet.get('Total Current Liabilities')

    # Non-current assets and liabilities
    summary['non_current_assets'] = balance_sheet.get('Total Non Current Assets', balance_sheet.get('Non-Current Assets'))
    summary['non_current_liabilities'] = balance_sheet.get('Total Non Current Liabilities', balance_sheet.get('Non-Current Liabilities'))

    # Retained earnings and other metrics
    summary['retained_earnings'] = balance_sheet.get('Retained Earnings')
    summary['working_capital'] = summary['current_assets'] - summary['current_liabilities'] if summary['current_assets'] and summary['current_liabilities'] else None
    summary['debt_to_equity_ratio'] = summary['total_liabilities'] / summary['total_shareholder_equity'] if summary['total_liabilities'] and summary['total_shareholder_equity'] else None
    summary['current_ratio'] = summary['current_assets'] / summary['current_liabilities'] if summary['current_assets'] and summary['current_liabilities'] else None
    summary['quick_ratio'] = (summary['current_assets'] - balance_sheet.get('Inventory', 0)) / summary['current_liabilities'] if summary['current_assets'] and summary['current_liabilities'] else None
    return summary
    

def summarize_cashflow(cashflow):
    summary = {}

    summary['net_operating_cash_flow'] = cashflow.get('Net Income From Continuing Ops')
    summary['changes_in_working_capital'] = cashflow.get('Change To Netincome')

    summary['capital_expenditures'] = cashflow.get('Capital Expenditures')
    summary['investments'] = cashflow.get('Investments')

    summary['dividends_paid'] = cashflow.get('Dividends Paid')
    summary['net_borrowings'] = cashflow.get('Net Borrowings')

    if summary['net_operating_cash_flow'] is not None and summary['capital_expenditures'] is not None:
        summary['free_cash_flow'] = summary['net_operating_cash_flow'] + summary['capital_expenditures']

    return summary

def summarize_earnings(earnings):
    summary = {
        'average_eps_estimate': earnings['EPS Estimate'].mean(),
        'average_reported_eps': earnings['Reported EPS'].mean(),
        'average_surprise_pct': earnings['Surprise(%)'].mean(),
        'latest_eps_estimate': earnings['EPS Estimate'].iloc[-1],
        'latest_reported_eps': earnings['Reported EPS'].iloc[-1],
        'latest_surprise_pct': earnings['Surprise(%)'].iloc[-1],
        'positive_surprises': earnings[earnings['Surprise(%)'] > 0].shape[0],
        'negative_surprises': earnings[earnings['Surprise(%)'] < 0].shape[0]
    }

    return summary

def summarize_quarterlyfin(quarterlyfin):

    quarterlyfin = quarterlyfin.T

    summary = {}
    
    summary = {
        'diluted_net_income': quarterlyfin['Diluted NI Availto Com Stockholders'],
        'net_income': quarterlyfin['Net Income'],
        'gross_profit': quarterlyfin['Gross Profit'],
        'operating_income': quarterlyfin['Operating Income'],
        'pretax_income': quarterlyfin['Pretax Income'],
        'total_revenue': quarterlyfin['Total Revenue']
    }
    return summary




def load_news_data(ticker):
    api_key = 'k5V7qUFxdSWWoKrxwoFqFlbxS90HLhyesf7QxWkC'
    
    url = f'https://api.marketaux.com/v1/news/all?symbols={ticker}&language=en&api_token={api_key}'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])  # Directly return the list of articles
        else:
            print(f"Error fetching data: {response.status_code}")
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def preprocess_data(news_data):
    processed_data = []
    for article in news_data:
        if isinstance(article, dict) and 'description' in article:
            clean_text = re.sub(r'<[^>]+>', '', article['description'])
            article['clean_description'] = clean_text  # Add cleaned description to the article dictionary
            processed_data.append(article)  # Append the entire article dictionary
        else:
            print(f"Warning: Unexpected format in article {article}")
    return processed_data


def feature_engineering(news_data, nlp_model, ner_model, summarization_model, transformer_model, transformer_tokenizer):
    features = []
    dates = []
    
    for article in news_data:
        description = article.get('description', '')

        sentiment = nlp_model(description)
        sentiment_score = 1 if sentiment[0]['label'] == 'POSITIVE' else 0

        entities = ner_model(description)
        entity_count = len(entities) if entities else 0

        summary = summarization_model(description)
        summary_length = len(summary[0]['summary_text']) if summary else 0

        transformer_sentiment = transformer_predict(description, transformer_model, transformer_tokenizer)

        combined_feature = {
            'sentiment': sentiment_score,
            'entity_count': entity_count,
            'summary_length': summary_length,
            'transformer_sentiment': transformer_sentiment
        }
        features.append(combined_feature)

    return pd.DataFrame(features)


def train_predictive_model(X_train, y_train):
    """
    Train a RandomForestRegressor model.
    """
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def load_transformer_model(model_name='bert-base-uncased'):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def preprocess_for_transformer(text, tokenizer, max_length=512):
    return tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

def transformer_predict(text, model, tokenizer):
    inputs = preprocess_for_transformer(text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1].item()  


def load_market_data():
    def fetch_index_data(index_ticker):
        index = yf.Ticker(index_ticker)
        historical_data = index.history(period="5y") 
        info = index.info

        return historical_data, info

    sp500_data, sp500_info = fetch_index_data("^GSPC")
    nasdaq_data, nasdaq_info = fetch_index_data("^IXIC")
    dowjones_data, dowjones_info = fetch_index_data("^DJI")

    return {
        "sp500": {"historical_data": sp500_data, "info": sp500_info},
        "nasdaq": {"historical_data": nasdaq_data, "info": nasdaq_info},
        "dowjones": {"historical_data": dowjones_data, "info": dowjones_info},
    }

  
def preprocess_market_industry_data(raw_data):
    processed_data = {}
    
    # Example of preprocessing
    for key, value in raw_data.items():
        historical_data = value["historical_data"]
        processed_data[f"{key}_daily_returns"] = historical_data['Close'].pct_change()
        processed_data[f"{key}_info"] = value["info"]

    return processed_data

def feature_engineering_market_industry(data):
    features = {}
    
    # Example of feature engineering
    for key in ['sp500', 'nasdaq', 'dowjones', 'stock']:
        if f"{key}_daily_returns" in data:
            daily_returns = data[f"{key}_daily_returns"]
            features[f"{key}_volatility"] = daily_returns.std() * (252 ** 0.5)

    # Add more features as needed
    return features


def main(ticker):

    market_data = load_market_data()
    print('Market data loaded.')

    # Preprocess market and industry data
    processed_market_data = preprocess_market_industry_data(market_data)
    print('Market data preprocessed.')

    # Feature engineering on market and industry data
    market_features = feature_engineering_market_industry(processed_market_data)
    
    
    print('Market features engineered.')
    raw_financial_data = load_financial_data(ticker)
    financial_summary = create_financial_summary(raw_financial_data)
    financial_summary_df = pd.json_normalize(financial_summary)
    news_data = load_news_data(ticker)
    processed_news_data = preprocess_data(news_data)
    print(financial_summary)
    print('--------------RAW ABOVE---------------------------------------------------------')
    print(financial_summary_df)


    # Instantiate NLP pipelines
    sentiment_model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    summarization_model = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
    transformer_model, transformer_tokenizer = load_transformer_model()

    news_features = feature_engineering(processed_news_data, sentiment_model, ner_model, summarization_model, transformer_model, transformer_tokenizer)

    combined_data = {
        "financial_summary": financial_summary,
        "news_analysis": news_features.to_dict(orient='records'),
        "market_data": market_data,
        "processed_market_data": processed_market_data,
        "market_features": market_features
    }
    print('---------------------------COMBINED DATA_________________________')
    print(combined_data)

    return combined_data

