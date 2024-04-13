import streamlit as st
import requests
import json
from openai import OpenAI

# Retrieve API Key and URL from Streamlit secrets
api_key = st.secrets["openai_api_key"]
api_url = st.secrets["api_url"]

client = OpenAI(api_key=api_key)

def fetch_portfolio_data(risk_level):
    """Fetch portfolio data from API based on risk profile."""
    url = f"{api_url}?riskprofile={risk_level}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to fetch data: {str(e)}"}

def interpret_risk_profile(risk_level):
    if 1 <= risk_level <= 3:
        return "Conservative"
    elif 4 <= risk_level <= 6:
        return "Moderate"
    elif 7 <= risk_level <= 10:
        return "Aggressive"
    else:
        return "Unknown"

def advisor_profile(data, user_query=None):
    """Generate investment advice using GPT model based on JSON data and user query."""
    if data is None or 'error' in data:
        return data.get('error', "Failed to parse JSON data from the API.")
    if not user_query:  # First advice request
        user_input = json.dumps(data)
    else:  # Subsequent user query
        user_input = user_query
        
    system_message = 'You are an investment advisor. Analyse the data and provide holistic advice in short, simple words to a customer, based on the JSON data.'
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            temperature=1,
            max_tokens=510,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"An error occurred while generating advice: {str(e)}"

# Streamlit UI Setup
st.title("Investment Risk Profile Advisor")
risk_level = st.slider("Select your risk profile on a scale of 1 to 10:", min_value=1, max_value=10, value=5)

if 'data' not in st.session_state or st.button("Get Initial Investment Advice"):
    st.session_state['risk_category'] = interpret_risk_profile(risk_level)
    st.session_state['data'] = fetch_portfolio_data(risk_level)
    st.session_state['advisor_info'] = advisor_profile(st.session_state['data'])

st.subheader(f"Your risk profile: {st.session_state['risk_category']}")
if 'error' in st.session_state['data']:
    st.error(st.session_state['data']['error'])
else:
    st.markdown(st.session_state['advisor_info'])

user_query = st.text_input("Ask your advisor a follow-up question:")
if st.button("Submit Query"):
    if user_query:
        response = advisor_profile(st.session_state['data'], user_query=user_query)
        st.session_state['advisor_info'] = response
        st.markdown(response)
    else:
        st.error("Please enter a question before submitting.")
