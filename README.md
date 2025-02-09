# Langchain PDF Chatbot

## Overview
Langchain PDF Chatbot is a Streamlit-based application that enables users to interact with PDF documents using natural language queries. It leverages OpenAI embeddings, FAISS vector search, and a conversational memory to provide contextual responses based on the contents of the uploaded PDF.

## Features
- Upload and process PDF documents.
- Extract and split text into manageable chunks for efficient querying.
- Store document embeddings in a FAISS vector database.
- Answer user queries based on document context and chat history.
- Maintain conversation history for better contextual understanding.

## Technologies Used
- **Streamlit**: Web framework for interactive UI.
- **PyPDF2**: Extracts text from PDFs.
- **Langchain**: Manages embeddings, vector stores, and LLM interactions.
- **OpenAI API**: Provides language model capabilities.
- **FAISS**: Efficient similarity search for document retrieval.
- **dotenv**: Loads environment variables.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-repo/langchain-pdf-chatbot.git
   cd langchain-pdf-chatbot
   ```

2. **Create a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key:**
   - Create a `.env` file in the project root.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. **Run the application:**
   ```sh
   streamlit run app.py
   ```

2. **Upload a PDF:**
   - Click on the file uploader and select a PDF document.

3. **Ask Questions:**
   - Type a query related to the document’s content.
   - Receive AI-generated responses based on the extracted text and chat history.

## Future Improvements
- Support for multiple PDFs.
- Integration with additional LLM providers.
- Enhanced document parsing and structured data extraction.
- UI/UX improvements for better user experience.

## License
This project is licensed under the MIT License.
