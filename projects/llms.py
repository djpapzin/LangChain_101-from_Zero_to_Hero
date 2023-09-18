from langchain.llms import OpenAIChat
from dotenv import load_dotenv

def run_llms(custom_prompt=None):
    load_dotenv()

    # Call the LLM
    llm = OpenAIChat(model="gpt-3.5-turbo", temperature=0.9)

    # The Prompt
    if not custom_prompt:
        custom_prompt = input("Enter your prompt (or press Enter for the default prompt): ")
        if not custom_prompt:
            custom_prompt = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."

    return llm(custom_prompt)

if __name__ == "__main__":
    result = run_llms()
    print(result)
