from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
try:
    from src.doc_retriver import langchain_docs_retriever
except Exception:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    from doc_retriver import langchain_docs_retriever

import os

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Better API key management (use environment variables or config file)
os.environ["GOOGLE_API_KEY"] = "AIzaSyD__Jvq_nu41rDuSk_9uSt1MiYMKv0Nsy4"
os.environ["TAVILY_API_KEY"] = "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"

# Initialize LLM with error handling
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,  # More deterministic for RAG
    )
except Exception as e:
    print(f"Error initializing LLM: {e}")
    raise

# Initialize Tavily with better configuration
tavily = TavilySearch(
    max_results=3,  # Reduced for better focus
    topics="general",
    country="india",
    search_depth="advanced"  # Better search quality
)

@tool
def websearch(query: str) -> str:
    """
    Search the web for the given query.

    Args:
        query (str): The search query.

    Returns:
        str: The search results.
    """
    try:
        response = tavily.invoke(query)
        # Format the response better
        if isinstance(response, list):
            formatted_results = []
            for item in response:
                if isinstance(item, dict):
                    title = item.get('title', 'No title')
                    content = item.get('content', 'No content')
                    formatted_results.append(f"Title: {title}\nContent: {content}\n")
            return "\n".join(formatted_results)
        return str(response)
    except Exception as e:
        return f"Error in web search: {str(e)}"

# Register available tools for the agent
tools = [
    websearch,
    langchain_docs_retriever,
]

def grade_document(state) -> Literal["generate", "rewrite"]:
    """
    Determine whether document is relevant to question or not

    Args:
        state (messages): current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """
    class Grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(
            description="Relevance score: 'yes' or 'no'",
            regex="^(yes|no)$"  # Ensure only yes/no responses
        )

    try:
        llm_bind_tool = llm.with_structured_output(Grade)

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question.

                Here is the retrieved document:
                {context}

                Here is the user question: {question}

                If the document contains keywords or semantic meaning related to the user question, grade it as relevant.
                Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.

                Consider the document relevant if:
                - It directly answers the question
                - It contains related concepts or information
                - It provides context that helps answer the question

                Score: """,
            input_variables=["context", "question"],
        )

        chain = prompt | llm_bind_tool

        messages = state["messages"]
        if not messages:
            return "rewrite"
        
        last_message = messages[-1]
        question = messages[0].content

        # Handle different message types
        if hasattr(last_message, 'content'):
            docs = last_message.content
        else:
            docs = str(last_message)

        score_result = chain.invoke({"context": docs, "question": question})
        score = score_result.binary_score.lower()

        print(f"Document grading score: {score}")

        if score == "yes":
            return "generate"
        else:
            return "rewrite"
            
    except Exception as e:
        print(f"Error in grade_document: {e}")
        return "generate"  # Default to generate on error

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state.
    
    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    messages = state["messages"]
    print("In agent state")
    

    try:
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(messages)
        print(f"Agent response: {response}")
        
        # Fix the key name bug: "message:" -> "messages"
        return {"messages": [response]}
        
    except Exception as e:
        print(f"Error in agent: {e}")
        error_msg = AIMessage(content=f"I encountered an error: {str(e)}")
        return {"messages": [error_msg]}

def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    messages = state["messages"]
    question = messages[0].content

    try:
        # More focused rewrite prompt
        msg = [
            HumanMessage(
                content=f"""Look at the input and try to reason about the underlying semantic intent/meaning.

Here is the initial question:
{question}

The previous search didn't return relevant results. Please reformulate this question to be:
1. More specific and focused
2. Using different keywords
3. Breaking down complex questions into simpler parts

Formulate an improved question:"""
            )
        ]

        response = llm.invoke(msg)
        print(f"Rewritten query: {response.content}")
        
        # Fix the key name bug: "message" -> "messages"
        # Replace the original question with the rewritten one
        new_message = HumanMessage(content=response.content)
        return {"messages": [new_message]}
        
    except Exception as e:
        print(f"Error in rewrite: {e}")
        return {"messages": [messages[0]]}  # Return original question

def generate(state):
    """
    Generate a response based on the current state.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with generated response
    """
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    try:
        # Handle different message types
        if hasattr(last_message, 'content'):
            context = last_message.content
        else:
            context = str(last_message)

        # Improved generation prompt
        prompt = PromptTemplate(
            template="""You are an AI assistant for question-answering tasks. Use the following retrieved context to answer the question.

Question: {question}

Context: {context}

Instructions:
- Provide a clear, accurate answer based on the context
- If the context doesn't contain enough information, say so clearly
- Keep the answer concise but comprehensive (3-5 sentences maximum)
- Cite specific information from the context when relevant

Answer:""",
            input_variables=["question", "context"]
        )

        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        response = chain.invoke({"question": question, "context": context})
        print(f"Generated response: {response}")
        
        # Fix the key name bug and return proper message format
        ai_message = AIMessage(content=response)
        return {"messages": [ai_message]}
        
    except Exception as e:
        print(f"Error in generate: {e}")
        error_response = AIMessage(content="I apologize, but I encountered an error while generating a response.")
        return {"messages": [error_response]}

# Build the graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("agent", agent)
builder.add_node("generate", generate)
builder.add_node("rewrite", rewrite)

# Fix: Create ToolNode with the correct tools
retrieve = ToolNode(tools)  # Use all tools, not undefined 'retriever_tool'
builder.add_node("retrieve", retrieve)  # Fix typo: "retrive" -> "retrieve"

# Add edges
builder.add_edge(START, "agent")

# Conditional edges from agent
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",  # Fix typo: "retrive" -> "retrieve"
        END: END,
    },
)

# Conditional edges from retrieve
builder.add_conditional_edges(
    "retrieve",
    grade_document,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

# Final edges
builder.add_edge("generate", END)
builder.add_edge("rewrite", "agent")

# Compile the graph
graph = builder.compile()

# Test the workflow with better error handling
def test_workflow():
    import pprint
    
    inputs = {
        "messages": [
            HumanMessage(content="how to setup conversation memory in langraph?"),
        ]
    }
    
    try:
        print("Starting workflow...")
        for output in graph.stream(inputs):
            for key, value in output.items():
                print(f"\n=== Output from node '{key}' ===")
                if 'messages' in value and value['messages']:
                    latest_message = value['messages'][-1]
                    if hasattr(latest_message, 'content'):
                        print(f"Content: {latest_message.content}")
                    else:
                        pprint.pprint(latest_message)
                else:
                    pprint.pprint(value)
                print("=" * 50)
    except Exception as e:
        print(f"Workflow error: {e}")
        import traceback
        traceback.print_exc()

def export_graph():
    """Export the compiled graph"""
    return graph

# Uncomment to test
if __name__ == "__main__":
    test_workflow()