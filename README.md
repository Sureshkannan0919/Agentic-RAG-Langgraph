# Agentic RAG - LangGraph-based Retrieval Augmented Generation

A sophisticated Retrieval Augmented Generation (RAG) system built with LangGraph that combines document retrieval, web search, and intelligent query processing to provide accurate answers to user questions.

## 🎯 Overview

This project implements an agentic RAG system that:
- Retrieves relevant information from LangChain documentation
- Performs web searches when needed using Tavily
- Grades document relevance and rewrites queries for better results
- Uses Google's Gemini 2.0 Flash model for generation
- Visualizes the workflow as a state graph

## 🏗️ Architecture

The system follows a graph-based workflow with these key components:

```
__start__ → agent → retrieve → grade_document
                ↓              ↓         ↓
               __end__    generate    rewrite → agent
                           ↓
                        __end__
```

### Key Components

1. **Agent Node**: Main orchestrator that decides which tools to use
2. **Retrieve Node**: Fetches documents from LangChain docs and web search
3. **Grade Document**: Evaluates relevance of retrieved documents
4. **Generate Node**: Creates final response based on relevant context
5. **Rewrite Node**: Reformulates queries when documents aren't relevant

## 🚀 Features

- **Multi-source Retrieval**: Combines local document search with web search
- **Intelligent Grading**: Evaluates document relevance before generation
- **Query Rewriting**: Automatically improves queries for better results
- **Vector Database**: Uses ChromaDB with HuggingFace embeddings
- **Graph Visualization**: Mermaid and ASCII diagram generation
- **Error Handling**: Robust error handling throughout the pipeline

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agentic_rag
```

2. Install required dependencies:
```bash
pip install langchain langgraph langchain-google-genai langchain-tavily
pip install langchain-community chromadb sentence-transformers
pip install pandas pickle-mixin
```

3. Set up API keys:
```bash
export GOOGLE_API_KEY="your-google-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

## 📋 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Google Generative AI API key for Gemini model
- `TAVILY_API_KEY`: Tavily search API key for web searches

### File Paths
- `chroma_db/`: ChromaDB vector database storage
- `docs_list.pkl`: Cached document embeddings
- `langchain_data.csv`: Source URLs for document loading (update path in `src/doc_retriver.py`)

## 🎮 Usage

### Basic Usage

Run the main workflow:
```bash
python main.py
```

### Testing the Retriever

Test document retrieval independently:
```bash
cd src
python doc_retriver.py
```

### Visualizing the Graph

Generate workflow visualizations:
```bash
python visualize_graph.py
```

This creates:
- `langgraph_mermaid.png` - Visual diagram (requires pygraphviz)
- `langgraph_mermaid.mmd` - Mermaid diagram text
- `langgraph_ascii.txt` - ASCII representation

### Custom Queries

Modify the test query in `main.py`:
```python
inputs = {
    "messages": [
        HumanMessage(content="Your question here"),
    ]
}
```

## 🔧 Tools Available

### Document Retriever (`langchain_docs_retriever`)
- Searches pre-indexed LangChain documentation
- Uses ChromaDB with sentence-transformers embeddings
- Returns top 5 most relevant document chunks

### Web Search (`websearch`)
- Powered by Tavily Search API
- Configured for India region with advanced search depth
- Returns top 3 formatted web results

## 🏃‍♂️ Workflow Steps

1. **User Query**: Input question is processed by the agent
2. **Tool Selection**: Agent decides whether to search docs, web, or both
3. **Document Retrieval**: Relevant documents are fetched
4. **Relevance Grading**: Documents are evaluated for relevance
5. **Response Generation**: If relevant, answer is generated
6. **Query Rewriting**: If not relevant, query is reformulated and retried

## 📁 Project Structure

```
agentic_rag/
├── main.py                 # Main workflow implementation
├── src/
│   └── doc_retriver.py    # Document retrieval and vector store
├── visualize_graph.py     # Graph visualization utilities
├── chroma_db/             # ChromaDB vector database
├── docs_list.pkl          # Cached documents
├── langgraph_mermaid.mmd  # Mermaid diagram
└── langgraph_mermaid.png  # Visual diagram
```

## 🎛️ Configuration Options

### LLM Configuration
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,  # Deterministic outputs
)
```

### Search Configuration
```python
tavily = TavilySearch(
    max_results=3,
    topics="general",
    country="india",
    search_depth="advanced"
)
```

### Vector Store Configuration
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

## 🔍 Example Interaction

```
Query: "How to setup conversation memory in LangGraph?"

1. Agent analyzes query
2. Calls langchain_docs_retriever tool
3. Retrieves relevant LangChain documentation
4. Grades documents as relevant
5. Generates comprehensive answer about LangGraph memory setup
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure environment variables are set correctly
2. **Import Errors**: Check if all dependencies are installed
3. **Vector DB Issues**: Delete `chroma_db/` folder to recreate from scratch
4. **Visualization Errors**: Install `pygraphviz` for PNG generation

### Error Handling

The system includes comprehensive error handling:
- Tool execution failures are caught and logged
- API errors return graceful error messages
- Fallback responses when generation fails

## 🚦 Performance Tips

1. **Vector DB**: Pre-built database loads much faster than rebuilding
2. **Query Optimization**: Specific queries work better than broad ones
3. **Context Length**: Adjust chunk size in document splitter if needed
4. **Temperature**: Keep at 0 for consistent RAG responses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [LangChain](https://github.com/langchain-ai/langchain) ecosystem
- Powered by Google Gemini and Tavily Search APIs
- Vector embeddings by HuggingFace Sentence Transformers

---

For questions or issues, please open a GitHub issue or contact the maintainer.
