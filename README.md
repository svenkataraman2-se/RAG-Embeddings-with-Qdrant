1. Follow the step by step instructions to run the project
2. Download the repository to your windows directory and open with VS code.
3. Download Ollama for windows from https://ollama.com/download and run ollama
4. pull and download any transformer, embedding model from huggingface.io
5. run llama3.1 from ollama using ollama run llama3.1 from git bash ( check for any newer open source ollama models)
6. using docker desktop for windows download QDrant vector DB image
7. run the QDrant vector db using the docker command : docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
8. Check if you can access Qdrant UI using http://localhost:6333/dashboard#/welcome
9. Set up python virtual environment using the below commands
    *     uv venv - creates a virtual env for downloading libraries for the app.
    *      Running the code using VS code will activate the virtual env.  manually **source .venv/Scripts/activate**
    *     download python app dependencies -
    *     uv pip install qdrant_client
    *     uv pip install PyMuPDF
  
10. Run the app via vs code and follow the prompts.

