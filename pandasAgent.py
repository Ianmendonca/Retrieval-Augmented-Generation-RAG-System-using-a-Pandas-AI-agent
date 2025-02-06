from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

import pandas as pd
import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set up OpenAI API Key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Load the Iris dataset
# iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
iris = pd.read_csv("E:\Masters\Semester 4\LSTM\Enery\Energy.csv")

# Initialize LangChain agent with the DataFrame
chat = ChatOpenAI(model_name='gpt-4', temperature=0.0)
agent = create_pandas_dataframe_agent(chat, iris, verbose=True, allow_dangerous_code=True)

# Streamlit UI
st.title("Dataset Analysis with AI")

# User input
user_query = st.text_input("Ask something about the dataset (e.g., 'Show a scatter plot of the data'): ")

if st.button("Submit"):
    with st.spinner("Thinking..."):
        response = agent.run(user_query)
        st.success(response)
        st.pyplot(plt)  # Show the latest Matplotlib figure

