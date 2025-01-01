import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

def load_openai_api_key():
    """
    Load the OpenAI API key from the environment file.
    """
    dotenv_path = "openai.env"
    load_dotenv(dotenv_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(f"Unable to retrieve OPENAI_API_KEY from {dotenv_path}. Ensure the key is correctly set.")
    return openai_api_key

def process_text(text):
    """
    Process and split the input text into manageable chunks.
    """
    # Split text into chunks using CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks into embeddings to create a knowledge base
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("📄 PDF Summarizer")
    st.write("**Created by Piyush Singhal**")
    st.divider()

    # Load OpenAI API key
    try:
        os.environ["OPENAI_API_KEY"] = load_openai_api_key()
    except ValueError as e:
        st.error(str(e))
        return

    # File upload section
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        try:
            pdf_reader = PdfReader(pdf)
            # Extract text from PDF
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Process the extracted text into a knowledge base
            st.info("Processing the uploaded PDF...")
            knowledge_base = process_text(text)

            # Pre-defined query for summarization
            query = (
                "Summarize the content of the uploaded PDF file in approximately 3-5 sentences. "
                "Focus on capturing the main ideas and key points discussed in the document. "
                "Use your own words and ensure clarity and coherence in the summary."
            )

            # Perform similarity search and generate summary
            docs = knowledge_base.similarity_search(query)
            openai_model = "gpt-3.5-turbo-16k"
            llm = ChatOpenAI(model=openai_model, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            st.info("Generating the summary...")
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                st.write(f"API Cost: {cost}")

            # Display the summary
            st.subheader('Summary Results:')
            st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

if __name__ == '__main__':
    main()
