import streamlit as st

def main():
    st.title("LangChain 101 Projects")
    

    # Create a dropdown menu for project selection
    project = st.selectbox(
        "Select a project to run:",
        ["agents", "chains", "llms", "memory", "tools", "vector_store"]
    )

    # Display the description for the selected project
    project_descriptions = {
        "agents": "Demonstrates the creation of an AI agent that leverages Google Search as a tool to answer questions about current events.",
        "chains": "Illustrates the creation of a chain using the OpenAI language model to suggest company names based on a given product.",
        "llms": "A simple script to demonstrate the usage of the OpenAI language model to suggest a personalized workout routine.",
        "memory": "Showcases a conversation chain with memory, allowing for continuous interactions.",
        "tools": "Demonstrates the usage of the OpenAI language model for text summarization and integrates Google Search API as a tool for the AI agent.",
        "vector_store": "Demonstrates the creation and usage of the DeepLake vector store with the OpenAI language model and embeddings."
    }

    st.write(project_descriptions[project])

    # Placeholder text for user input based on the selected project
    project_placeholders = {
        "agents": "E.g., 'What's the latest news about the Mars rover?'",
        "chains": "E.g., 'I want to start a company that sells organic tea.'",
        "llms": "E.g., 'Tell me a joke.'",
        "memory": "E.g., 'Hello!'",
        "tools": "E.g., 'What's the latest news about the Mars rover? Then please summarize the results.'",
        "vector_store": "E.g., 'When was Napoleon born?'"
    }

    user_input = st.text_input("Enter your input:", placeholder=project_placeholders[project])

    if st.button("Run"):
        with st.spinner('Processing your input...'):
            if project == "agents":
                import agents
                output = agents.run_agents(user_input)
            elif project == "chains":
                import chains
                output = chains.run_chains(user_input)
            elif project == "llms":
                import llms
                output = llms.run_llms(user_input)
            elif project == "memory":
                import memory
                output = memory.run_conversation(user_input)
            elif project == "tools":
                import tools
                output = tools.run_tools(user_input)
            elif project == "vector_store":
                import vector_store
                output = vector_store.run_vector_store(user_input)
            else:
                st.write("Invalid choice!")
            
            st.write(output)

if __name__ == "__main__":
    main()