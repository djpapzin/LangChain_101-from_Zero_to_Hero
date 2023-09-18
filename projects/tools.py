# Import necessary modules and classes
from langchain.llms import OpenAIChat
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve Google API key and Custom Search Engine ID from environment variables
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
GOOGLE_CSE_ID = os.environ["GOOGLE_CSE_ID"]

# Initialize the OpenAI model with the specified parameters
llm = OpenAIChat(model="gpt-3.5-turbo", temperature=0)

# Define a prompt template for text summarization
prompt = PromptTemplate(
    input_variables=["query"],
    template="Write a summary of the following text: {query}"
)

# Create an LLMChain instance for text summarization
summarize_chain = LLMChain(llm=llm, prompt=prompt)

# Initialize the Google Search API wrapper
search = GoogleSearchAPIWrapper()

# Define the tools that the agent will use
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events"
    ),
    Tool(
       name='Summarizer',
       func=summarize_chain.run,
       description='useful for summarizing texts'
    )
]

# Initialize an agent that uses the defined tools
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

def run_tools(query):
    """
    Function to run the agent with a given query.
    """
    # Run the agent with the provided query
    response = agent(query)
    
    # Return the agent's response
    return response['output']

if __name__ == "__main__":
    # Get input from the user
    user_query = input("Enter your query: e.g. What's the latest news about the Mars rover? Then please summarize the results. ")
    
    # Run the tools with the user's query and print the result
    print(run_tools(user_query))
