import json
import os
import sys
import boto3
import streamlit as st
import time

## We will be using Titan Emeddings Model to gerenate the embeddings for the given text

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


# Vetor Embedding and Vector store
from langchain_community.vectorstores import FAISS

## LLm Modle 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


## Bedrock Client
bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


## Constants
MAX_TOKENS = 500

## Data Ingestion

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    #splitter = RecursiveCharacterTextSplitter()
    #documents = loader.load_documents()
    #texts = [splitter.split_text(document.text) for document in documents]
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    #texts = [text_splitter.split_text(document.text) for document in documents]
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and Vector Store
def get_vector_store(docs):
    #docs = data_ingestion()
    #embeddings = bedrock_embeddings.embed_documents(docs)
    #vector_store = FAISS()
    #vector_store.add_embeddings(embeddings)
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    
def get_claude_llm():
    ## Create Antropic Claude LLM
    llm = Bedrock(model_id = "anthropic.claude-v2:1", client=bedrock, model_kwargs = {'max_tokens_to_sample': MAX_TOKENS})
    return llm

def get_llma3_llm():
    ## Create LLMA3  LLM
    llm = Bedrock(model_id = "meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs = {'max_gen_len': MAX_TOKENS})
    return llm  


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
#print(PROMPT)
def get_response_llm(llm,vectorstore_faiss,query,response_placeholder):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    # Print the context and question values
    context = vectorstore_faiss.as_retriever().get_relevant_documents(query)[0].page_content
    print(f"Context variable =: {context}")
    print(f"Question variable = : {query}")
    
    answer=qa({"query":query})
    if 'result' in answer:
        result = answer['result']
        if "I do not have enough context" in result or "I apologize" in result:
            response_placeholder.write("The model could not find relevant information to answer your question.")
        else:
            # Simulate streaming by updating the response incrementally
            for i in range(0, len(result), 50):
                response_placeholder.write(result[:i+50])
                time.sleep(0.1)  # Simulate delay
    else:
        response_placeholder.write("No valid response from the model.")

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                print(f"Documents: {docs}")  # Debugging line
                if not docs:
                    st.error("No documents found. Please check the data directory.")
                else:
                    get_vector_store(docs)
                    st.success("Done")
    response_placeholder = st.empty()
    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_claude_llm()
            get_response_llm(llm, faiss_index, user_question, response_placeholder)
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_llma3_llm()
            print(llm)
            get_response_llm(llm, faiss_index, user_question,response_placeholder)
            st.success("Done")

if __name__ == "__main__":
    main()
