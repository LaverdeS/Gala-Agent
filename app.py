import os
import logging
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from retriever import guest_info_tool
from tools import weather_info_tool, hub_stats_tool

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

# Generate the chat interface, and append the tools
tools = [guest_info_tool, weather_info_tool, hub_stats_tool]


llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

"""
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],  # HUGGINGFACEHUB_API_TOKEN
)
chat = ChatHuggingFace(llm=llm, verbose=True)
chat_with_tools = chat.bind_tools(tools)
"""

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],  # or chat_with_tools
    }

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

image_data = alfred.get_graph(xray=True).draw_mermaid_png()
with open("gala_agent_thought_process.png", "wb") as f:
    f.write(image_data)

messages = [HumanMessage(content="Marie")]  # Tell me about our guest named 'Lady Ada Lovelace'.
response = alfred.invoke({"messages": messages})

# Show the messages
for m in response['messages']:
    m.pretty_print()

final_answer = response['messages'][-1].content

print("🎩 Alfred's Response:")
print(final_answer)