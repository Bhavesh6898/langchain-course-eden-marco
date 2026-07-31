from operator import itemgetter
import os

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:"""
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LangChain Expressions Language.
    Manually retrieves documents, formats them and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support wothout additional code
    - Harder to compose with other chains
    - More verbose and error prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


# ================================================
# Imolementation 2: with LCEL (Langchain Expression Language) - Better Approach
# ================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch Processing: chain.batch() for multiple inputs
    - Type safety: Better integration with the LangChain's type system
    - Less code: More concise and readable
    - Reusable: chain can be saved, shared and composed with other chains
    - Better debugging: Langchain provides better observability tools
    """
    retrieval_chain=(
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template 
        | llm 
        | StrOutputParser()
    )
    return retrieval_chain



if __name__ == "__main__":
    print("retrieving...")

    # Query
    query = "what is pinecone in machine learning?"

    # ================================================
    # Option1: Use implementation without LCEL
    # ================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    # result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print("result_without_lcel")

    # ================================================
    # Option2: Use implementation with LCEL
    # ================================================
    print("\n" + "=" * 70)
    print("Impelementation 2: with LCEL - Better Approach")
    print("=" * 70)
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)