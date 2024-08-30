# RAG LangChain Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, which has been designed to interact with users based on the entire **Harry Potter** book series. The chatbot leverages embeddings from the Sentence Transformers library, stored in Chroma DB, to retrieve relevant content and generates conversational responses using the ChatGPT API. The backend is powered by FastAPI, and the user interface is implemented with Gradio.

## Features

- **Harry Potter Knowledge Base**: The chatbot has been trained on all Harry Potter books, enabling it to respond with contextually accurate information from the series.
- **Advanced Retrieval System**: Uses embeddings from the Sentence Transformers library to embed both the corpus and user prompts. The retriever fetches the most relevant responses from the embedded Harry Potter content based on a similarity threshold.
- **Conversational Response Generation**: The retrieved information is passed to the ChatGPT API, which formulates the response in a natural, conversational manner.
- **Persistent Conversation**: The chatbot maintains conversation history, allowing for continuous, context-aware interactions.
- **FastAPI Backend**: The application is served via a FastAPI backend, ensuring fast and efficient API handling.
- **Gradio User Interface**: The UI is designed with Gradio, providing an intuitive and user-friendly interface for interaction with the chatbot.

## Architecture

1. **Data Ingestion**:
   - All Harry Potter books were ingested and stored.
   - Each book was processed to create embeddings using the Sentence Transformers library.

2. **Chroma DB**:
   - The embeddings were stored in Chroma DB, a high-performance embedding storage solution.
   - Chroma DB is used to efficiently retrieve relevant text passages based on user prompts.

3. **Embedding and Retrieval**:
   - User prompts are embedded using the same Sentence Transformers model.
   - The embedded prompt is compared with the stored embeddings to retrieve the most relevant responses based on a predefined similarity threshold.

4. **Response Generation**:
   - The retrieved text is passed to the ChatGPT API.
   - ChatGPT formulates a response in a human-like conversational style.

5. **API and UI**:
   - A FastAPI backend handles the API requests and responses.
   - Gradio is used to create a web-based UI where users can interact with the chatbot.

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- Gradio
- Chroma DB
- Sentence Transformers
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AsawirShafiq/RAG-LangChain-chatbot.git
   cd RAG-LangChain-chatbot
   ```
2. Install requirments:
   ```bash
   pip install -r requirements.txt
   ```
## Usage

- **Start the FastAPI server** and interact with the chatbot via the Gradio UI.
- **Enter your questions or prompts** in the Gradio interface, and the chatbot will respond with information derived from the Harry Potter books.
- **Conversation history** is maintained, allowing for continuous, context-aware discussions.

