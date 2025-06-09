import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    """Format the documents for the custom RAG prompt."""
    return "\n\n".join([doc.page_content for i, doc in enumerate(docs)])

if __name__ == "__main__":
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as consise as possible.
    Always say "thanks for asking!" at the end of your answer.
    
    {context}

    Question: {question}

    Helpful answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template=template)


    print(" Retrieving...")

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    llm = ChatOllama(model="gemma3:1b")

    query = "what is Pinecone in machine learning?"

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrival_chain = create_retrieval_chain(
    #     retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    # ) | StrOutputParser()

    result = rag_chain.invoke(query)

    print(result)