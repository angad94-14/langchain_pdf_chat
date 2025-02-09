# make a file called app.py in the langchain_pdf_chat directory

from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain



def main():
    load_dotenv()
    st.set_page_config(page_title='Langchain PDF Chat', page_icon='📄')
    st.header('Chat with your PDF 💬' )

    # upload file
    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    # extract file text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200, # 200 characters overlap
            length_function=len
        )
        chunks = splitter.split_text(text)
        # st.write(chunks)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings) # create the vector store using openai embeddings

        # get user input
        user_question = st.text_input("Ask a question!")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            st.write(docs)
            # Extract text from Document objects
            context = "\n\n".join([doc.page_content for doc in docs])
            st.write(context)

            # Define Prompt Template
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
            )

            # Instantiate LLM
            llm = OpenAI()

            # Create the Stuff Documents Chain using the latest function
            chain = create_stuff_documents_chain(llm, prompt)

            # Invoke chain with formatted input
            result = chain.invoke({"context": docs, "question": user_question})

            # Display result
            st.write("Answer:", result)

    # print(os.getenv('OPENAI_API_KEY'))

if __name__ == '__main__':
    main()