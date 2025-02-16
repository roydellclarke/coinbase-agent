import json
import os
import sys
import time
import logging
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Langchain & LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent

# CDP Agentkit imports
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.tools import CdpTool
from pydantic import BaseModel, Field
from cdp import *

# ---------------- Configuration & Logging ----------------
wallet_data_file = "wallet_data.txt"
MAX_ITERATIONS = 3
TIMEOUT = 15
REQUEST_TIMEOUT = 20

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Agent Initialization ----------------
def initialize_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize the LLM.
    api_key = os.getenv("OPENAI_API_KEY")
    cdp_api_key_name = os.getenv("CDP_API_KEY_NAME")
    cdp_api_key_private_key = os.getenv("CDP_API_KEY_PRIVATE_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    if not cdp_api_key_name or not cdp_api_key_private_key:
        raise ValueError("CDP API key environment variables are not set")
        
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=api_key,
        timeout=TIMEOUT,
        request_timeout=REQUEST_TIMEOUT,
        max_retries=2,
        temperature=0.7
    )

    wallet_data = None
    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    values = {
        "cdp_api_key_name": cdp_api_key_name,
        "cdp_api_key_private_key": cdp_api_key_private_key
    }
    if wallet_data is not None:
        # Load the persisted wallet data.
        values["cdp_wallet_data"] = wallet_data

    # Initialize the CDP Agentkit wrapper.
    agentkit = CdpAgentkitWrapper(**values)
    # Persist (or update) the agent's wallet.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Get the CDP tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    # Store conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create the ReAct agent using the LLM and CDP tools.
    agent_executor = create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=(
            "You are a helpful agent that can interact onchain using the Coinbase Developer Platform Agentkit. "
            "You are empowered to interact onchain using your tools. If you ever need funds, you can request them from "
            "the faucet if you are on network ID `base-sepolia`. If not, you can provide your wallet details and request "
            "funds from the user. If someone asks you to do something you can't do with your currently available tools, "
            "you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit. Recommend they "
            "go to docs.cdp.coinbase.com for more information. Be concise and helpful with your responses. Refrain from "
            "restating your tools' descriptions unless it is explicitly requested."
        )
    )
    return agent_executor, config, tools

# Global initialization
AGENT_EXECUTOR, AGENT_CONFIG, CDP_TOOLS = initialize_agent()

# ---------------- Agent State Definition ----------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]   # Conversation history (accumulated)
    next_action: Optional[str]                    # Either "tools" or "end"
    tool_calls: Optional[List[Dict[str, Any]]]    # Structured tool call instructions (if any)
    iterations: int                               # Iteration counter to avoid infinite loops

# ---------------- Reasoner Node ----------------
def reasoner(state: AgentState) -> AgentState:
    """
    The reasoner node:
      1. Increments the iteration counter and forces termination after MAX_ITERATIONS.
      2. Invokes the agent (via its streaming interface) with the current conversation.
      3. Accumulates agent responses and checks for any tool-call instructions.
      4. Sets the next_action flag to "tools" if tool calls are detected, or "end" otherwise.
    """
    # Increment iterations.
    iterations = state.get("iterations", 0) + 1
    if iterations >= MAX_ITERATIONS:
        termination_message = AIMessage(content="Maximum iterations reached. Ending conversation.")
        return {
            "messages": [termination_message],
            "next_action": "end",
            "tool_calls": None,
            "iterations": iterations
        }
    state["iterations"] = iterations
    logger.info(f"Reasoner iteration {iterations} with {len(state['messages'])} message(s).")

    new_messages: List[BaseMessage] = []
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Invoke the agent using its streaming interface.
    # The agent_executor expects a dict with "messages" and a config.
    for chunk in AGENT_EXECUTOR.stream({"messages": state["messages"]}, AGENT_CONFIG):
        if "agent" in chunk:
            agent_msg = chunk["agent"]["messages"][0]
            new_messages.append(agent_msg)
        elif "tools" in chunk:
            tool_msg = chunk["tools"]["messages"][0]
            new_messages.append(tool_msg)
            # Try to parse tool call instructions from the tool message.
            try:
                parsed = json.loads(tool_msg.content)
                # Expecting parsed to be a list of tool call dicts.
                tool_calls = parsed
            except Exception as e:
                # If parsing fails, wrap the content in a default tool call structure.
                tool_calls = [{"tool": "cdp_tool", "input": tool_msg.content}]
    next_action = "tools" if tool_calls else "end"
    return {
        "messages": new_messages,
        "next_action": next_action,
        "tool_calls": tool_calls,
        "iterations": iterations
    }

# ---------------- Graph Setup ----------------
workflow = StateGraph(AgentState)
workflow.add_node("reasoner", reasoner)
workflow.add_node("tools", ToolNode(CDP_TOOLS))
workflow.add_conditional_edges(
    "reasoner",
    lambda state: state["next_action"],
    {
        "tools": "tools",  # If tool_calls exist, route to the Tools node.
        "end": END         # Otherwise, terminate.
    }
)
workflow.add_edge("tools", "reasoner")  # After tool execution, return to the reasoner.
workflow.set_entry_point("reasoner")
graph = workflow.compile()

# ---------------- Chat Mode Response Function ----------------
def handle_user_input(agent_executor, user_input: str) -> str:
    """
    Process user input through the agent and return the response.
    This function is used by both the chat mode and the Gradio interface.
    """
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "next_action": None,
        "tool_calls": None,
        "iterations": 0
    }
    
    # Get all messages from the final state
    final_state = graph.invoke(initial_state)
    all_messages = final_state["messages"]
    
    # Combine all AI messages into a single response
    response_parts = []
    for msg in all_messages:
        if isinstance(msg, AIMessage):
            response_parts.append(msg.content)
    
    # Return combined response or default message
    if response_parts:
        return "\n".join(response_parts)
    return "No response generated."

def run_chat_mode():
    """Run the agent interactively based on user input."""
    print("Starting chat mode... Type 'exit' to end.")
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() == "exit":
                break
            response = handle_user_input(AGENT_EXECUTOR, user_input)
            print("Response:", response)
            print("-------------------")
        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)

# ---------------- Autonomous Mode ----------------
def run_autonomous_mode(interval: int = 10):
    """Run the agent in autonomous mode at fixed time intervals."""
    print("Starting autonomous mode...")
    while True:
        try:
            thought = (
                "Be creative and do something interesting on the blockchain. "
                "Choose an action or set of actions and execute it that highlights your abilities."
            )
            response = handle_user_input(AGENT_EXECUTOR, thought)
            print("Response:", response)
            print("-------------------")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)

# ---------------- Mode Selection ----------------
def choose_mode() -> str:
    """Prompt the user to choose between chat mode and autonomous mode."""
    while True:
        print("\nAvailable modes:")
        print("1. chat    - Interactive chat mode")
        print("2. auto    - Autonomous action mode")
        choice = input("\nChoose a mode (enter number or name): ").lower().strip()
        if choice in ["1", "chat"]:
            return "chat"
        elif choice in ["2", "auto"]:
            return "auto"
        print("Invalid choice. Please try again.")

# ---------------- Main ----------------
def main():
    mode = choose_mode()
    if mode == "chat":
        run_chat_mode()
    elif mode == "auto":
        run_autonomous_mode()

if __name__ == "__main__":
    print("Starting Agent...")
    main()
