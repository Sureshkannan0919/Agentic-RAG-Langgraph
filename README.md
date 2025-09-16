## LangGraph-powered RAG Agent for LangChain Docs

An example agent built with LangGraph and LangChain that can answer questions using:
- Web search via Tavily
- A local Chroma vector database of LangChain documentation

The agent routes between tools, optionally rewrites queries for better retrieval, and generates concise answers.

### Features
- **Tooling**: `websearch` (Tavily) and `langchain_docs_retriever` (Chroma)
- **Routing**: Grades retrieved docs to choose generate vs rewrite
- **Visualization**: Mermaid/PNG/ASCII graph diagrams

### Project Structure
```text
.
├── main.py                    # Builds the LangGraph agent and workflow
├── src/
│   └── doc_retriver.py        # Chroma retriever builder over LangChain docs
├── visualize_graph.py         # Utilities to export/visualize the graph
├── chroma_db/                 # Persisted Chroma vector store (if present)
├── docs_list.pkl              # Cached documents (optional, auto-created)
├── langgraph_mermaid.mmd      # Mermaid diagram (generated)
└── langgraph_mermaid.png      # Mermaid PNG (generated)
```

### Requirements
- Python 3.10+
- pip

Recommended packages (install all):
```bash
pip install \
  langgraph langchain langchain-core langchain-community \
  langchain-google-genai langchain-tavily \
  chromadb sentence-transformers tiktoken \
  pandas beautifulsoup4 requests pydantic
```

Optional for visualization:
```bash
pip install pygraphviz ipython
```

Note: On some systems, `pygraphviz` requires Graphviz system libraries. If installation fails, skip PNG export and use Mermaid/ASCII outputs instead.

### Environment Variables
Set your API keys before running:
```bash
export GOOGLE_API_KEY="<your_google_api_key>"
export TAVILY_API_KEY="<your_tavily_api_key>"
```

Important: `main.py` currently sets placeholder keys inline. For security, remove or override those lines and use environment variables instead.

### Data for the Retriever
`src/doc_retriver.py` loads an existing Chroma DB if `chroma_db/` exists. If not, it can build one from a CSV file with a column named `URL`:

- Configure the CSV path by updating `CSV_PATH` in `src/doc_retriver.py` (defaults to `/home/suresh/projects/langchain_data.csv`).
- On first run without an existing DB, it will:
  - Fetch pages listed in the CSV
  - Split and embed the documents
  - Persist a Chroma DB to `chroma_db/`
  - Cache the raw docs in `docs_list.pkl`

If you already have `chroma_db/` in this repository, the retriever will load it directly without needing the CSV.

### Quick Start
1) Create and activate a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:
```bash
pip install \
  langgraph langchain langchain-core langchain-community \
  langchain-google-genai langchain-tavily \
  chromadb sentence-transformers tiktoken \
  pandas beautifulsoup4 requests pydantic
```

3) Set API keys:
```bash
export GOOGLE_API_KEY="<your_google_api_key>"
export TAVILY_API_KEY="<your_tavily_api_key>"
```

4) Run the example workflow:
```bash
python main.py
```

You should see the agent stream through nodes and print responses.

### Programmatic Usage
You can import and run the graph yourself:
```python
from langchain_core.messages import HumanMessage
from main import graph

inputs = {"messages": [HumanMessage(content="how to setup conversation memory in langraph?")]} 
for output in graph.stream(inputs):
    print(output)
```

### Visualizing the Graph
Export Mermaid/PNG/ASCII representations of the workflow:
```bash
python visualize_graph.py
```

Artifacts will be created in the project root:
- `langgraph_mermaid.mmd`
- `langgraph_mermaid.png` (requires `pygraphviz`)
- `langgraph_ascii.txt`

### Troubleshooting
- **Missing packages**: Ensure all required packages are installed. Some components (e.g., `pygraphviz`) are optional.
- **Sentence-transformers download issues**: If model download fails, try: `pip install --upgrade pip setuptools wheel` and ensure internet access.
- **Tiktoken errors**: Install `tiktoken`, and if building from source fails, upgrade `pip`.
- **Chroma persistence issues**: Delete `chroma_db/` and re-run to rebuild.
- **API key errors**: Verify `GOOGLE_API_KEY` and `TAVILY_API_KEY` are exported in your shell.

### Security Notes
- Never commit real API keys. Prefer environment variables over hardcoding.
- Review `main.py` and remove inline key assignments before production use.

### License
Add your license here (e.g., MIT).

