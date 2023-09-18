# Import necessary modules and classes
from langchain.llms import OpenAIChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAIChat model with the specified parameters
llm = OpenAIChat(model="gpt-3.5-turbo", temperature=0)

# Create a conversation chain with the specified LLM, verbosity, and memory type
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

def run_conversation(user_input=None):
    """
    Function to run a conversation loop between the user and the bot.
    The loop continues until the user types 'exit'.
    """
    # Start the conversation and provide instructions to the user
    print("Starting the conversation. Type 'exit' to end the conversation.")
    
    # If user_input is provided, use it directly. Otherwise, prompt the user for input.
    if user_input is None:
        user_input = input("You: ")
    
    # Check if the user wants to exit the conversation
    if user_input.lower() == 'exit':
        return "You ended the conversation."
    
    # Get the response from the bot using the conversation chain
    response = conversation({'input': user_input})
    
    # Return the bot's response
    return f"Bot: {response['response']}"

# If the script is run as the main program, execute the run_conversation function
if __name__ == "__main__":
    run_conversation()