import pandas as pd
from openai import OpenAI
import os 
from dotenv import load_dotenv
import json

load_dotenv('key.env')

OpenAI.api_key=os.getenv('OPENAI_API_KEY')
client = OpenAI()

def generate_analysis_prompt(score, combined_data):
    prompt = f"Based on the following market data, the market health score was determined to be {score}. Please analyze and explain how this score reflects the current state of the market:\n"
    prompt += json.dumps(combined_data, indent=2)
    return prompt

def get_market_analysis(score, combined_data, model="gpt-4"):
    prompt = generate_analysis_prompt(score, combined_data)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    if response.choices:
        response_text = response.choices[0].message.content.strip()
    else:
        response_text = "No response received."
    return response_text


def generate_eco_analysis_prompt(score, combined_data):
    prompt = f"Based on the following economic data, the economic health score was determined to be {score}. Please analyze and explain how this score reflects the current state of the economy:\n"
    prompt += json.dumps(combined_data, indent=2)
    return prompt

def get_eco_analysis(score, combined_data, model="gpt-4"):
    prompt = generate_eco_analysis_prompt(score, combined_data)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    if response.choices:
        response_text = response.choices[0].message.content.strip()
    else:
        response_text = "No response received."
    return response_text