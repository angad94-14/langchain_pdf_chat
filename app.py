import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


def initialize_memories():
    if "main_memory" not in st.session_state:
        st.session_state.main_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    if "summary_memory" not in st.session_state:
        st.session_state.summary_memory = ConversationSummaryMemory(
            llm=OpenAI(),
            memory_key="summary_history",
            return_messages=True
        )


def create_research_chains(llm):
    # Chain for extracting key concepts
    concept_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Extract key concepts from: {context}\nFor question: {question}\nKey concepts:"
    )
    concept_chain = LLMChain(llm=llm, prompt=concept_prompt)

    # Chain for finding supporting evidence
    evidence_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Find supporting evidence for this question: {question}\nIn context: {context}\nEvidence:"
    )
    evidence_chain = LLMChain(llm=llm, prompt=evidence_prompt)

    # Chain for generating counter-arguments
    counter_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Generate potential counter-arguments for this question: {question}\nUsing context: {context}\nCounter-arguments:"
    )
    counter_chain = LLMChain(llm=llm, prompt=counter_prompt)

    return concept_chain, evidence_chain, counter_chain


def main():
    load_dotenv()
    st.set_page_config(page_title='Advanced Research Assistant', page_icon='🔬')
    st.header('Research Assistant with Multi-Perspective Analysis 🔬')

    # Initialize memories
    initialize_memories()

    # Initialize knowledge base
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    # File upload - support multiple PDFs
    pdfs = st.file_uploader("Upload research papers", type=["pdf"], accept_multiple_files=True)

    if pdfs and st.session_state.knowledge_base is None:
        with st.spinner('Processing documents...'):
            text = ''
            for pdf in pdfs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = splitter.split_text(text)

            embeddings = OpenAIEmbeddings()
            st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)
            st.success("Documents processed! Ready for research.")

    # Research query input
    research_question = st.text_input("Enter your research question:")

    if research_question and st.session_state.knowledge_base:
        # Retrieve relevant documents
        docs = st.session_state.knowledge_base.similarity_search(research_question)

        # Convert docs to string for context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Initialize LLM and chains
        llm = OpenAI(temperature=0.7)
        concept_chain, evidence_chain, counter_chain = create_research_chains(llm)

        # Create parallel processing pipeline with independent chains
        research_chain = RunnableParallel(
            concepts=lambda x: concept_chain.run(
                context=x["context"],
                question=x["question"]
            ),
            evidence=lambda x: evidence_chain.run(
                context=x["context"],
                question=x["question"]
            ),
            counter_arguments=lambda x: counter_chain.run(
                context=x["context"],
                question=x["question"]
            )
        )

        # Process research with proper input structure
        with st.spinner('Analyzing...'):
            result = research_chain.invoke({
                "context": context,
                "question": research_question
            })

        # Save to memories
        st.session_state.main_memory.save_context(
            {"input": research_question},
            {"output": f"Analysis Results:\n{result}"}
        )

        st.session_state.summary_memory.save_context(
            {"input": research_question},
            {"output": f"Key Findings: {result['concepts']}"}
        )

        # Display results in organized tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Key Concepts",
            "Supporting Evidence",
            "Counter Arguments",
            "Research History"
        ])

        with tab1:
            st.markdown("### Key Concepts Identified")
            st.write(result["concepts"])

        with tab2:
            st.markdown("### Supporting Evidence")
            st.write(result["evidence"])

        with tab3:
            st.markdown("### Counter Arguments")
            st.write(result["counter_arguments"])

        with tab4:
            st.markdown("### Research History")
            st.markdown("#### Detailed History")
            messages = st.session_state.main_memory.load_memory_variables({})["chat_history"]
            for msg in messages:
                st.text(f"{msg.type}: {msg.content}")

            st.markdown("#### Summary")
            summary = st.session_state.summary_memory.load_memory_variables({})["summary_history"]
            st.write(summary)


if __name__ == '__main__':
    main()