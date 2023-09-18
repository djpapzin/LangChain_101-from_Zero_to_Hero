from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAIChat
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Retrieve the values
ACTIVELOOP_TOKEN = os.environ["ACTIVELOOP_TOKEN"]
my_activeloop_org_id = os.environ['MY_ACTIVELOOP_ORG_ID']

# instantiate the LLM and embeddings models
llm = OpenAIChat(model="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# create Deep Lake dataset
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)

# create a RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

# create an agent that uses the RetrievalQA chain as a tool
tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_vector_store(question=None):
    if not question:
        question = input("Please enter your question: e.g. When was Napoleon born? ")
    response = agent.run(question)
    return response

if __name__ == "__main__":
    print(run_vector_store())
