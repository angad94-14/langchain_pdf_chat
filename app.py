import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory  # Stores conversation history

def main():
    # Load environment variables (e.g., API keys)
    load_dotenv()

    # Set up Streamlit page with a title and an icon
    st.set_page_config(page_title='Langchain PDF Chat', page_icon='📄')
    st.header('Chat with your PDF 💬')

    # Initialize session state for knowledge base (FAISS index) and chat history
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None  # Stores the FAISS vector database
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Stores past interactions

    # Allow the user to upload a PDF file
    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Process the PDF only once and store embeddings persistently
    if pdf is not None and st.session_state.knowledge_base is None:
        pdf_reader = PdfReader(pdf)
        text = ''

        # Extract text from each page of the uploaded PDF
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Split extracted text into smaller chunks to optimize retrieval
        splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,  # Define the chunk size
            chunk_overlap=200,  # Overlap between chunks to retain context
            length_function=len
        )
        chunks = splitter.split_text(text)  # Perform text chunking

        # Convert text chunks into embeddings and store them in FAISS for fast similarity search
        embeddings = OpenAIEmbeddings()
        st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Notify the user that the document has been processed
        st.write("PDF processed! You can now ask questions.")

    # Get user input (query) from Streamlit's text input field
    user_question = st.text_input("Ask a question!")

    if user_question and st.session_state.knowledge_base:
        # Retrieve relevant document chunks from FAISS using similarity search
        docs = st.session_state.knowledge_base.similarity_search(user_question)

        # Define a prompt template for the LLM to generate answers
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

        # Instantiate OpenAI's LLM model
        llm = OpenAI()

        # Create a document-based LLM chain using the prompt and model
        chain = create_stuff_documents_chain(llm, prompt)

        # Prepare chat history as a string to pass as context
        history = "\n".join(st.session_state.chat_history)

        # 🔹 FIX: Pass `docs` directly instead of converting to a string
        result = chain.invoke({"history": history, "context": docs, "question": user_question})

        # Append the user query and the model's response to the chat history
        st.session_state.chat_history.append(f"User: {user_question}")
        st.session_state.chat_history.append(f"Bot: {result}")

        # Display conversation history (last 10 interactions)
        st.write("### Conversation History:")
        for msg in st.session_state.chat_history[-10:]:
            st.write(msg)

# Ensure the `main` function runs when executing the script
if __name__ == '__main__':
    main()
