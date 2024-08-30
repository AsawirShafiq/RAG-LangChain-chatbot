
import os
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable import RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)



model= ChatOpenAI(model='gpt-4o', openai_api_key=api_key)
model1 = ChatOpenAI(model='gpt-4o', openai_api_key=api_key)

# Define a custom embeddings class
class CustomEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=False).tolist()

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(books_dir)
print(persistent_directory)

# Checking if the Chroma store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Making sure the file actually exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError("Directory does not exist")

    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n------Document Chunks Info---------")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = CustomEmbeddings(model)
    print("\n-----------Created embeddings------------")

    # Create the vector store and persist it
    print("\n-------Creating and persisting vector store-----")
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persistent_directory)
    print("Finished creating vector store")
else:
    print("Vector store already exists")

