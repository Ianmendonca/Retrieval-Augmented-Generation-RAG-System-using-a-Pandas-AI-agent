import os
import openai
import requests
from fpdf import FPDF
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI

# access the environment key for OPENAI API key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
movie_url = os.getenv("URL")

#xtract movie script
def extractScript(url):
    '''
    using the url extract the movie script and save it in a pdf file
    '''
    response = requests.get(url)

    if response.status_code == 200:
        script_text = response.text
    else:
        raise Exception(f"Failed to fetch the script. Status code: {response.status_code}")

    # Step 2: Initialize PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Step 3: Add text to the PDF
    for line in script_text.splitlines():
        pdf.multi_cell(0, 10, line)

    # Step 4: Save the PDF
    output_path = "bee_movie_script.pdf"
    pdf.output(output_path)

#Fetch the text content
pdf_path = 'bee_movie_script.pdf'  # Replace with the actual PDF file path
pdf_text = ""

# check if movie script exists, if not extract using the movie url
if not os.path.exists(pdf_path):
    extractScript(movie_url)
    
# Open and read the PDF
reader = PdfReader(pdf_path)
for page in reader.pages:
    pdf_text += page.extract_text()

# Step 2: Split the text into chunks
# Adjust the chunk size based on the PDF content (e.g., by paragraphs or sentences)
def chunk_text_by_length(text, max_tokens=512):
    tokens = text.split()  # Split by whitespace (this is a rough tokenization)
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        # If the chunk exceeds the token limit, start a new chunk
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add the last chunk if it contains remaining tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

script_chunks = chunk_text_by_length(pdf_text)
# Create document objects
documents = [Document(page_content=chunk) for chunk in script_chunks]

# Initialize vector store with OpenAI embeddings
vector_store = FAISS.from_documents(documents, OpenAIEmbeddings())

def sentiment_analysis_tool(query):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in analyzing character sentiments in movies."},
            {"role": "user", "content": query}
        ]
    )
    return response['choices'][0]['message']['content']

# create a tool to retrieve the text
def script_retrieval_tool(query):
    # Retrieve the most relevant script chunk
    docs = vector_store.similarity_search(query, k=1)
    return docs[0].page_content 


# Define tools for the agent
tools = [
    # tool 1 fot sentiment analysis
    Tool(
        name="Sentiment Analysis",
        func=sentiment_analysis_tool,
        description="Analyze the sentiment of the following quote whether it is positive, negative, neutral, or mixed and explain how the character feelings"
    ),
    # tool 2 for summarization and sentence completion
    Tool(
        name="Script Retrieval",
        func=script_retrieval_tool,
        description="Use this tool to answer questions about the Bee Movie by retrieving relevant context and do not repeat the question if relevant text is found else return sorry, I couldnt find relevant context"
    ),
]

# Define the system prompt

# Initialize the LLM with the system prompt
llm = ChatOpenAI(
    model="gpt-4", 
    temperature=0,
    max_tokens = 250
)
# initialize agents
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)

def bee_movie_agent(query):
    response = agent.run(query)
    return response

# Streamlit UI
st.title("Movie Sentiment Analysis Chatbot")

# User input
user_query = st.text_input("Ask your question:")
if st.button("Submit"):
    with st.spinner("Thinking..."):
        response = agent.run(user_query)
        st.success(response) 