import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import json


def initialize_chat_history():
    """Initialize chat history and insurance-specific state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_policy_sections" not in st.session_state:
        st.session_state.current_policy_sections = {}
    if "highlighted_terms" not in st.session_state:
        st.session_state.highlighted_terms = set()


def create_insurance_chain(llm):
    """Create a specialized chain for insurance document understanding."""
    insurance_prompt = PromptTemplate(
        input_variables=["context", "question", "customer_profile"],
        template="""You are an experienced insurance advisor helping a customer understand their policy and highlighting
        technical details from the policy that match their query. 

        Customer Profile: {customer_profile}
        Question: {question}
        Policy Context: {context}

        Please provide a clear, simple explanation that:
        1. Answers the question directly
        2. Highlights any important coverage limits or exclusions or technical terms that are part of the policy context
        3. Uses everyday language instead of insurance jargon
        4. Includes relevant examples when helpful
        5. Suggests follow-up questions the customer might want to ask

        Response:"""
    )
    return LLMChain(llm=llm, prompt=insurance_prompt)


def display_chat_message(msg, is_user=False, has_alert=False):
    """Display a chat message with insurance-specific styling."""
    style = """
        padding: 10px;
        border-radius: 15px;
        margin: 5px;
        max-width: 70%;
        """

    if is_user:
        align = "flex-end"
        bg_color = "#007AFF"
        color = "white"
    else:
        align = "flex-start"
        bg_color = "#F0F0F0"
        color = "black"

    if has_alert:
        bg_color = "#FFE4E1"  # Light red for important warnings

    st.write(
        f'<div style="display: flex; justify-content: {align};">'
        f'<div style="{style} background-color: {bg_color}; color: {color};">'
        f'{msg}</div></div>',
        unsafe_allow_html=True
    )


def initialize_customer_profile():
    """Create or update customer profile information."""
    st.sidebar.markdown("### Your Profile")

    # Get or initialize customer profile
    if "customer_profile" not in st.session_state:
        st.session_state.customer_profile = {
            "policy_type": "",
            "life_stage": "",
            "coverage_concerns": [],
            "risk_factors": []
        }

    # Collect profile information
    policy_type = st.sidebar.selectbox(
        "Type of Insurance",
        ["Health", "Life", "Auto", "Home", "Disability"],
        key="policy_type_select"
    )

    life_stage = st.sidebar.selectbox(
        "Life Stage",
        ["Single", "Married", "Family with Children", "Retired"],
        key="life_stage_select"
    )

    coverage_concerns = st.sidebar.multiselect(
        "Coverage Priorities",
        ["Comprehensive Coverage", "Low Deductibles", "Family Protection",
         "Asset Protection", "Retirement Planning"],
        key="concerns_select"
    )

    # Update profile in session state
    st.session_state.customer_profile.update({
        "policy_type": policy_type,
        "life_stage": life_stage,
        "coverage_concerns": coverage_concerns
    })


def main():
    load_dotenv()
    st.set_page_config(page_title='Insurance Policy Assistant', page_icon='📋')
    st.header('Interactive Insurance Policy Guide 📋')

    # Initialize states and profiles
    initialize_chat_history()
    initialize_customer_profile()

    # Initialize knowledge base
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    # File upload for policy documents
    with st.sidebar:
        st.markdown("### Your Policy Documents")
        pdfs = st.file_uploader(
            "Upload your insurance policy documents",
            type=["pdf"],
            accept_multiple_files=True
        )

    if pdfs and st.session_state.knowledge_base is None:
        with st.spinner('Analyzing your policy documents...'):
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
            st.success("✅ Documents processed! Ask any questions about your policy.")

    # Initialize LLM with insurance-specific configuration
    llm = OpenAI(temperature=0.7)

    # Create main interface with tabs
    tab1, tab2, tab3 = st.tabs([
        "Policy Chat",
        "Coverage Summary",
        "Important Terms"
    ])

    with tab1:
        st.markdown("### Ask About Your Policy")

        # Display suggested questions based on profile
        st.markdown("#### Suggested Questions")
        if st.session_state.customer_profile["policy_type"] == "Health":
            st.write("• What is my deductible and how does it work?")
            st.write("• Are my preferred doctors in-network?")
            st.write("• How does my prescription drug coverage work?")
        elif st.session_state.customer_profile["policy_type"] == "Life":
            st.write("• What are my beneficiary options?")
            st.write("• How does the death benefit work?")
            st.write("• Are there any exclusions I should know about?")

        # Chat interface
        chat_container = st.container()

        with chat_container:
            for message in st.session_state.chat_history:
                display_chat_message(
                    message["content"],
                    message["is_user"],
                    message.get("has_alert", False)
                )

        # Query input
        with st.form(key="chat_form", clear_on_submit=True):
            query = st.text_input("Ask a question about your policy:", key="query_input")
            submit_button = st.form_submit_button("Ask")

        if submit_button and query and st.session_state.knowledge_base:
            # Display user question
            with chat_container:
                display_chat_message(query, is_user=True)

            # Generate response
            with st.spinner('Finding information...'):
                docs = st.session_state.knowledge_base.similarity_search(query)
                context = "\n\n".join([doc.page_content for doc in docs])

                insurance_chain = create_insurance_chain(llm)
                response = insurance_chain.run(
                    context=context,
                    question=query,
                    customer_profile=json.dumps(st.session_state.customer_profile)
                )

                # Check for important warnings or exclusions
                has_alert = any(word in response.lower()
                                for word in ["exclusion", "warning", "limitation", "not covered"])

                with chat_container:
                    display_chat_message(response, is_user=False, has_alert=has_alert)

            # Update chat history
            st.session_state.chat_history.extend([
                {"content": query, "is_user": True},
                {"content": response, "is_user": False, "has_alert": has_alert}
            ])

    with tab2:
        st.markdown("### Your Coverage Overview")
        if st.session_state.knowledge_base:
            # Generate and display policy summary
            summary_prompt = "Provide a simple summary of the key coverage points and limits."
            docs = st.session_state.knowledge_base.similarity_search(summary_prompt)
            context = "\n\n".join([doc.page_content for doc in docs])

            summary_chain = create_insurance_chain(llm)
            summary = summary_chain.run(
                context=context,
                question=summary_prompt,
                customer_profile=json.dumps(st.session_state.customer_profile)
            )

            st.write(summary)

    with tab3:
        st.markdown("### Important Terms & Definitions")
        if st.session_state.knowledge_base:
            # Extract and explain important insurance terms
            terms_prompt = "List and explain important insurance terms from the policy."
            docs = st.session_state.knowledge_base.similarity_search(terms_prompt)
            context = "\n\n".join([doc.page_content for doc in docs])

            terms_chain = create_insurance_chain(llm)
            terms = terms_chain.run(
                context=context,
                question=terms_prompt,
                customer_profile=json.dumps(st.session_state.customer_profile)
            )

            st.write(terms)


if __name__ == '__main__':
    main()