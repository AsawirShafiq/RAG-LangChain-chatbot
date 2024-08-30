# from langchain_community.vectorstores import Chroma
# import os
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.embeddings.base import Embeddings
# from sentence_transformers import SentenceTransformer
# from langchain.chat_models import ChatOpenAI
# from langchain_core.messages import (
#     AIMessage,
#     BaseMessage,
#     FunctionMessage,
#     HumanMessage,
#     SystemMessage,
#     ToolMessage,
# )
# #deifne the persistant directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# db_dir = os.path.join(current_dir, "db")
# persistent_directory = os.path.join(db_dir,  "chroma_db_with_metadata")

# class CustomEmbeddings(Embeddings):
#     def __init__(self, model):
#         self.model = model

#     def embed_documents(self, texts):
#         return self.model.encode(texts, convert_to_tensor=False).tolist()

#     def embed_query(self, text):
#         return self.model.encode(text, convert_to_tensor=False).tolist()

# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = CustomEmbeddings(model)

# db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# query = "Who kills Jiriya?"

# #Retrieve relevant documents based on the query

# retriever = db.as_retriever(
#   search_type="similarity_score_threshold",
#   search_kwargs={"k":3, "score_threshold":0.2}
# )

# relevant_docs = retriever.invoke(query)

# #display
# print("------relavant stuff--------")
# for i, doc in enumerate(relevant_docs,1):
#   print(f"document {i}:\n{doc.page_content}\n")
#   if doc.metadata:
#     print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# combined_input =(
#   "Here are some documents that might help answer the question:" +
#   query +
#   "\n\nRelevant Documents:\n"+
#   "\n\n".join([doc.page_content for doc in relevant_docs]) +
#   "\n\nPlease provide an answer based only on the provided documents."
# )
# api_key = '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'
# model = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

# #define the message
# messages = [
#   SystemMessage(content = "You are a helpful assistant."),
#   HumanMessage(content=combined_input)
# ]

# result = model.invoke(messages)

# print("\n------generated Response-----")
# print(result.content)


api_key = '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'
import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# # Set up directories and embeddings
# current_dir = os.path.dirname(os.path.abspath(__file__))
# db_dir = os.path.join(current_dir, "db")
# persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# class CustomEmbeddings(Embeddings):
#     def __init__(self, model):
#         self.model = model

#     def embed_documents(self, texts):
#         return self.model.encode(texts, convert_to_tensor=False).tolist()

#     def embed_query(self, text):
#         return self.model.encode(text, convert_to_tensor=False).tolist()

# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = CustomEmbeddings(model)

# # Load the existing Chroma vector store
# db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# # Set up the retriever
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 3, "score_threshold": 0.1}
# )

# # Initialize the LLM model
# llm_model = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

# # Conversation loop
# conversation_history = []

# while True:
#     # Get the user query
#     query = input("You: ")

#     if query.lower() in ["exit", "quit"]:
#         print("Ending conversation.")
#         break

#     # Retrieve relevant documents based on the query
#     relevant_docs = retriever.invoke(query)

#     # Display the relevant documents (optional)
#     print("------Relevant Documents--------")
#     for i, doc in enumerate(relevant_docs, 1):
#         print(f"Document {i}:\n{doc.page_content}\n")
#         if doc.metadata:
#             print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

#     # Combine input for the LLM model
#     combined_input = (
#         "Here are some documents that might help answer the question:\n\n" +
#         query +
#         "\n\nRelevant Documents:\n" +
#         "\n\n".join([doc.page_content for doc in relevant_docs]) +
#         "\n\nPlease provide an answer based only on the provided documents."
#     )

#     # Add the query and retrieved information to the conversation history
#     conversation_history.append(HumanMessage(content=query))
#     conversation_history.append(SystemMessage(content=combined_input))

#     # Generate the response from the LLM
#     result = llm_model.invoke(conversation_history)

#     # Display the LLM's response
#     print("\n------Generated Response-----")
#     print(result.content)

#     # Add the LLM's response to the conversation history
#     conversation_history.append(result)

from fastapi import FastAPI, Request
from pydantic import BaseModel


app = FastAPI()

# Initialize models and vector store (similar to your script)
class CustomEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=False).tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = CustomEmbeddings(model)

# Set up directories and embeddings
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

llm_model = ChatOpenAI(model="gpt-4", openai_api_key=api_key)
conversation_history = []

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("message")

    if not query:
        return {"response": "Please provide a message."}

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.2}
    )
    relevant_docs = retriever.invoke(query)

    # Combine input for the LLM model
    combined_input = (
        f"Here are some documents that might help answer the question:\n\n{query}\n\n"
        f"Relevant Documents:\n" +
        "\n\n".join([doc.page_content for doc in relevant_docs]) +
        "\n\nPlease provide an answer based only on the provided documents. If there are questions that are not in the scope of these documents just answer by saying Sorry I do not know."
    )

    # Add the query and retrieved information to the conversation history
    conversation_history.append(HumanMessage(content=query))
    conversation_history.append(SystemMessage(content=combined_input))

    # Generate the response from the LLM
    result = llm_model.invoke(conversation_history)

    # Add the LLM's response to the conversation history
    conversation_history.append(result)

    return {"response": result.content}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

