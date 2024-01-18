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

def aggregate_news_features(sp500_features, nsd_features, dow_features):
    combined_news_features = pd.concat([sp500_features, nsd_features, dow_features]).mean()
    return combined_news_features.to_dict()

def combine_market_news_data(market_features, combined_news_features):

    combined_data = {**market_features, **combined_news_features}
    return combined_data

def load_market_data_news():
    market_data = load_market_data()
    print('Market data loaded.')
    processed_market_data = preprocess_market_industry_data(market_data)
    print('Market data preprocessed.')
    market_features = feature_engineering_market_industry(processed_market_data)


    sentiment_model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    summarization_model = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
    transformer_model, transformer_tokenizer = load_transformer_model()

    sp500_news = load_news_data('^GSPC')
    processed_sp500 = preprocess_data(sp500_news)
    sp500_features = feature_engineering(processed_sp500, sentiment_model, ner_model, summarization_model, transformer_model, transformer_tokenizer)

    NSD_news = load_news_data('^IXIC')
    processed_NSD = preprocess_data(NSD_news)
    NSD_features = feature_engineering(processed_NSD, sentiment_model, ner_model, summarization_model, transformer_model, transformer_tokenizer)

    dow_news = load_news_data('^DJI')
    processed_dow = preprocess_data(dow_news)
    dow_features = feature_engineering(processed_dow, sentiment_model, ner_model, summarization_model, transformer_model, transformer_tokenizer)

    combined_news_features = aggregate_news_features(sp500_features, NSD_features, dow_features)
    combined_data = combine_market_news_data(market_features, combined_news_features)

    print(combined_data)
    return combined_data


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


def generate_chat_prompt(combined_data):
    print(combined_data)
    prompt = "Based on the following market and news data, rate the current market health on a scale of 0-10, in the 10th so 0.0-10.0. Lower is a bad rating higher is a good rating:\n"
    prompt += json.dumps(combined_data, indent=2)
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

    

