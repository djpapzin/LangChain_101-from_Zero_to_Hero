from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.chains import LLMChain
from dotenv import load_dotenv

def run_chains(product_name=None):
    load_dotenv()

    llm = OpenAIChat(model="gpt-3.5-turbo", temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    
    if product_name is None:
        product_name = input("Enter your product name: ")

    return chain.run(product_name)

if __name__ == "__main__":
    product_name = input("Enter your product name: ")
    result = run_chains(product_name)
    print(result)
