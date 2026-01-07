Follow the step by step instructions to run the project
Download the source and open with VS code.
Download Ollama for windows from https://ollama.com/download and run ollama
pull and download a model from huggingface.io
run llama3.1 from ollama using ollama run llama3.1 from git bash
using docker desktop download QDrant vector DB image
run the QDrant vector db using the docker command : docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
Access Qdrant UI using http://localhost:6333/dashboard#/welcome

set up python virtual environment using the below commands
* uv venv - creates a virtual env for downloading libraries for the app.
* download python app dependencies - uv pip install fitz, qdrant_client, ollama

