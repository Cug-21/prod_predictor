from openai import OpenAI
import os 
from dotenv import load_dotenv

load_dotenv('key.env')

OpenAI.api_key=os.getenv('OPENAI_API_KEY')
client = OpenAI()

def generate_chat_prompt(data):
    print(data)

    prompt = "Analyze the following data for a short term (2 weeks) and mid term (6 months) and long term (2 years) health assessment of the company using provided data such as news sentiment analysis & quantitaitive data. Feel free to use the forcastes we provide from the currect 2023 quarterly finanicals and currect index prices and trends. \n\n"
    prompt += "Financial Summary:\n"
    for key, value in data['financial_summary'].items():
        prompt += f"{key}: {value}\n"
    
    prompt += "\nNews Analysis:\n"
    for item in data['news_analysis']:
        description = item.get('description', 'No description available')
        sentiment = item.get('sentiment', 'No sentiment data')
        entity_count = item.get('entity_count', 'No entity count data')
        summary_length = item.get('summary_length', 'No summary length data')
        transformer_sentiment = item.get('transformer_sentiment', 'No transformer sentiment data')
        prompt += f"Description: {description}, Sentiment: {sentiment}, Entity Count: {entity_count}, Summary Length: {summary_length}, Transformer Sentiment: {transformer_sentiment}\n"
    
    prompt += "\nMarket Data Analysis:\n"
    for key in ['sp500', 'nasdaq', 'dowjones']:
        if key in data['market_features']:
            prompt += f"{key.upper()} Volatility: {data['market_features'][f'{key}_volatility']}\n"
    

    return prompt


def get_chat_response(data, model="gpt-4"):
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

    