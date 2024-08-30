
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
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

api_key = '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'

model= ChatOpenAI(model='gpt-4o', openai_api_key=api_key)
model1 = ChatOpenAI(model='gpt-4o', openai_api_key=api_key)

# messages = [
#   SystemMessage(content='Solve the following math problems. Solve only math problems'),
#   #HumanMessage(content='What is 23 times 2?'),
#   HumanMessage(content='What is the capital of Pakistan'),
#   #HumanMessage(content='What is 23 divided by 0?')
# ]

# result = model.invoke(messages)
# #print(result)
# print("content only:")
# print(result.content)

#Having a conversation with chatgpt but also remebering the history
# chat_history = []
# while True:
#   prompt = input("you: ")
#   if prompt.lower() == "exit":
#     break
#   chat_history.append(HumanMessage(content=prompt))

#   response = model.invoke(chat_history)
#   response = response.content
#   chat_history.append(AIMessage(content=response))

#   print(f"AI: {response}")

# print("___History___")
# print(chat_history)

#prompt example and Serial chaining example

# prompt_template = ChatPromptTemplate.from_messages([
#   ("system", "You are a storyteller who tells stories about {topics}."),
#   ("human", "Tell me {story_count} stories."),
# ])

# messages = [
#    SystemMessage(content='Summarize the text'),
# ]

# def anothermodel(text):
#   print(text)
#   prompt_template = ChatPromptTemplate.from_messages([
#   ("system", "You are will summarize all that has been given."),
#   ("human", "{text}"),

#   ])
#   prompt = prompt_template.invoke({"text": text})
#   result = model1.invoke(prompt)
#   return result.content

# def furthermodel(text):
#   print(text)
#   prompt_template = ChatPromptTemplate.from_messages([
#   ("system", "You will extract only the titles of the text given"),
#   ("human", "{text}"),

#   ])
#   prompt = prompt_template.invoke({"text": text})
#   result = model1.invoke(prompt)
#   return result.content

# lower_output = RunnableLambda(lambda x: anothermodel(x))
# another_output = RunnableLambda(lambda x: furthermodel(x))

# chain = prompt_template | model | StrOutputParser() |lower_output| another_output

# result = chain.invoke({"topics": "lawyers", "story_count": 3})

# print(result)

#parallel branching
# sends a product, its features are xtracted by a model, a seperate model finds its cons and a seperate finds its pros.
# Another model takes both as input to turn the result into bulletpoints

# prompt_template = ChatPromptTemplate.from_messages([
#    ("system", "You will tell us all the features and specifications of any given topic."),
#    ("human", "Tell me the features of {product}."),
# ])

# def analyse_pros(text):

#    prompt_template = ChatPromptTemplate.from_messages([
#    ("system", "You will extract all the pros from the given prompt"),
#    ("human", "{text}"),

#    ])
#    prompt = prompt_template.invoke({"text": text})
#    result = model1.invoke(prompt)
#    print("From pros")
#    print(result.content)
#    return result.content
# def analyse_cons(text):

#    prompt_template = ChatPromptTemplate.from_messages([
#    ("system", "You will extract all the cons from the given prompt"),
#    ("human", "{text}"),

#    ])
#    prompt = prompt_template.invoke({"text": text})
#    result = model1.invoke(prompt)
#    print("From cons")
#    print(result.content)
#    return result.content

# def combine(text):

#    prompt_template = ChatPromptTemplate.from_messages([
#    ("system", "You will combine the given prompt in bullet points only"),
#    ("human", "{text}"),

#    ])
#    prompt = prompt_template.invoke({"text": text})
#    result = model1.invoke(prompt)
#    return result.content

# pros_branch = RunnableLambda(lambda x: analyse_pros(x))


# cons_branch = RunnableLambda(lambda x: analyse_cons(x))


# combine_branch = RunnableLambda(lambda x: combine(x))


# parallel = RunnableParallel(pros=pros_branch, cons=cons_branch)

# chain = prompt_template | model | StrOutputParser() | parallel | combine_branch

# result = chain.invoke({"product": "Iphone 12 mini"})

# print(result)

#Example of branching

# prompt_template = ChatPromptTemplate.from_messages([
#    ("system", "You will take review from users"),
#    ("human", "{review}"),
# ])

# def classify_positive(text):

#    prompt_template = ChatPromptTemplate.from_messages([
#    ("system", "You will act like a manager that's gotten a positive review and thank the customer."),
#    ("human", "{text}"),

#    ])
#    prompt = prompt_template.invoke({"text": text})
#    result = model1.invoke(prompt)
#    print("From positive")
#    print(result.content)
#    return result.content

# def classify_negative(text):

#    prompt_template = ChatPromptTemplate.from_messages([
#    ("system", "You will act like a manager that's gotten a negative review and apologies the customer."),
#    ("human", "{text}"),

#    ])
#    prompt = prompt_template.invoke({"text": text})
#    result = model1.invoke(prompt)
#    print("From negative")
#    print(result.content)
#    return result.content

# def classify_neutral(text):

#    prompt_template = ChatPromptTemplate.from_messages([
#    ("system", "You will act like a manager that's gotten a neutral review and tell customer that you look forward to further patronage."),
#    ("human", "{text}"),

#    ])
#    prompt = prompt_template.invoke({"text": text})
#    result = model1.invoke(prompt)
#    print("From Neutral")
#    print(result.content)
#    return result.content

# def classify(text):
#    print(text)
#    prompt_template = ChatPromptTemplate.from_messages([
#    ("system", "You will classify the given text into either Positive or Negative or Netural sentiment"),
#    ("human", "{text}"),

#    ])
#    prompt = prompt_template.invoke({"text": text})
#    result = model1.invoke(prompt)
#    print("classify")
#    print(result.content)
#    return result.content

# classification_lambda = RunnableLambda(lambda x: classify_positive(x) if "positive" in x.lower() else classify_negative(x) if "negative" in x.lower() else classify_neutral(x) if "neutral" in x.lower() else "Unclassified")
# print_branch = RunnableLambda(lambda x: print(x))

# classification = RunnableLambda(lambda x: classify(x))


# parallel = RunnableParallel(positive=classification_lambda, negative=classification_lambda, neutral = classification_lambda)

# chain = prompt_template | classification | StrOutputParser() |parallel

# result = chain.invoke({"review": "Your products are the worst ever!"})

#print(result)



import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

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

