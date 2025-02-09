import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.memory import ConversationBufferMemory


def get_memory():
    """Get or create memory object"""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
    return st.session_state.memory


def display_chat_history():
    """Display the chat history in a conversational format"""
    memory = get_memory()
    messages = memory.load_memory_variables({})["history"]

    # Create a container for chat history
    chat_container = st.container()

    with chat_container:
        for msg in messages:
            # Check if the message is a string (older format) or a Message object
            if isinstance(msg, str):
                st.write(msg)
            else:
                # For Message objects, check the type and display accordingly
                if msg.type == 'human':
                    st.write(f"👤 **You:** {msg.content}")
                else:
                    st.write(f"🤖 **Assistant:** {msg.content}")

        # Add a divider after the history
        if messages:
            st.divider()


def main():
    # Load environment variables
    load_dotenv()

    # Set up Streamlit page
    st.set_page_config(page_title='Langchain PDF Chat', page_icon='📄')
    st.header('Chat with your PDF')

    # Initialize memory at the start of the app
    memory = get_memory()

    # Initialize knowledge base in session state
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    # Upload PDF
    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Process the PDF and store embeddings persistently
    if pdf is not None and st.session_state.knowledge_base is None:
        pdf_reader = PdfReader(pdf)
        text = ''

        # Extract text from each page of the PDF
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Split text into chunks for better retrieval
        splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text)

        # Convert text into embeddings and store in FAISS
        embeddings = OpenAIEmbeddings()
        st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.write("PDF processed. You can now ask questions.")

    # Display chat history before the input box
    display_chat_history()

    # Get user query
    user_question = st.text_input("Ask a question!")

    if user_question and st.session_state.knowledge_base:
        # Retrieve relevant document chunks from FAISS
        docs = st.session_state.knowledge_base.similarity_search(user_question)

        # Define a prompt template
        prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=(
                "You are a helpful assistant. Use the chat history and the given context to answer the question.\n\n"
                "Chat History:\n{history}\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        )

        # Instantiate OpenAI LLM
        llm = OpenAI()

        # Function to retrieve conversation history
        def get_chat_history(_):
            return memory.load_memory_variables({})["history"]

        # Create a processing pipeline using RunnableParallel
        chain = (
                RunnableParallel(
                    history=RunnablePassthrough() | get_chat_history,
                    context=RunnablePassthrough(),
                    question=RunnablePassthrough()
                )
                | prompt
                | llm
        )

        # Invoke chain
        result = chain.invoke({"context": docs, "question": user_question})

        # Save conversation history
        memory.save_context(
            {"input": user_question},
            {"output": result}
        )

        # Clear the input box after sending (optional)
        st.empty()


if __name__ == '__main__':
    main()