from langchain.llms import OpenAIChat
from langchain.agents import AgentType 
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
import os
from dotenv import load_dotenv

def run_agents(user_query=None):
    # Load environment variables
    load_dotenv()
    
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    GOOGLE_CSE_ID = os.environ["GOOGLE_CSE_ID"]

    # Initialize the LLM and set the temperature to 0 for the precise answer. 
    llm = OpenAIChat(model="gpt-3.5-turbo", temperature=0)

    # Define the Google search wrapper
    search = GoogleSearchAPIWrapper()

    # The Tool object represents a specific capability or function the system can use.
    tools = [
        Tool(
            name = "google-search",
            func=search.run,
            description="useful for when you need to search google to answer questions about current events"
        )
    ]

    # create an agent that uses our Google Search tool
    agent = initialize_agent(tools, 
                             llm, 
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                             verbose=True,
                             max_iterations=6)

    # Capture user input for the prompt
    if user_query is None:
        user_query = input("Enter your query (e.g., 'What's the latest news about the Mars rover?'): ")

    # run the agent with the user's query
    response = agent(user_query)
    return response['output']

if __name__ == "__main__":
    user_query = input("Enter your query (e.g., 'What's the latest news about the Mars rover?'): ")
    result = run_agents(user_query)
    print(result)