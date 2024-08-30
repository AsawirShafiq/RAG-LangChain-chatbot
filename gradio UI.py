
import gradio as gr
import requests

# Define the function that will interact with the FastAPI app
def call_fastapi(message):
    # Replace this URL with the actual URL of your FastAPI app
    api_url = "http://localhost:8000/chat"

    # Make a POST request to the FastAPI app
    response = requests.post(api_url, json={"message": message})

    # Extract the response content
    if response.ok:
        return response.json().get("response", "No response from API")
    else:
        return f"Error: {response.status_code} - {response.text}"

# Define the Gradio interface
iface = gr.Interface(
    fn=call_fastapi,
    inputs="text",
    outputs="text",
    title="LangChain Chatbot Interface",
    description="This chabot contains all information about Harry Potter"
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch(share=True)
