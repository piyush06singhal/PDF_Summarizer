import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

def load_openai_api_key():
    dotenv_path = "openai.env"
    if not os.path.exists(dotenv_path):
        raise FileNotFoundError(f"File '{dotenv_path}' not found. Please ensure the file exists.")
    load_dotenv(dotenv_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(f"Unable to retrieve OPENAI_API_KEY from {dotenv_path}")
    return openai_api_key

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

def main():
    st.title("📄 PDF Summarizer")
    st.write("Created by Piyush Singhal")
    st.divider()

    try:
        os.environ["OPENAI_API_KEY"] = load_openai_api_key()
        st.success("API Key loaded successfully!")
    except Exception as e:
        st.error(f"Error: {e}")
        return

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        try:
            pdf_reader = PdfReader(pdf)
            # Text variable will store the pdf text
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

            if not text.strip():
                st.error("Unable to extract text from the uploaded PDF. Please check the document.")
                return

            st.info("Processing the text and creating the knowledge base...")
            knowledgeBase = process_text(text)
            st.success("Knowledge base created successfully!")

            query = "Summarize the content of the uploaded PDF file in approximately 3-5 sentences. Focus on capturing the main ideas and key points discussed in the document. Use your own words and ensure clarity and coherence in the summary."

            if query:
                docs = knowledgeBase.similarity_search(query)
                OpenAIModel = "gpt-3.5-turbo-16k"
                llm = ChatOpenAI(model=OpenAIModel, temperature=0.1)
                chain = load_qa_chain(llm, chain_type='stuff')

                st.info("Generating summary...")
                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)
                    st.write(f"OpenAI API Cost: {cost}")

                st.subheader('Summary Results:')
                st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return

if __name__ == '__main__':
    main()
