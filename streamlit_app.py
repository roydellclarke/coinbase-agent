import streamlit as st
from typing import Iterator
import time
from coinbase_agent import initialize_agent, handle_user_input

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor, st.session_state.agent_config, st.session_state.cdp_tools = initialize_agent()

def stream_response(response: str) -> Iterator[str]:
    """Stream the response with a typing effect."""
    full_response = []
    for word in response.split():
        full_response.append(word)
        yield " ".join(full_response)
        time.sleep(0.05)  # Adjust typing speed

# Page config
st.set_page_config(
    page_title="Base Blockchain AI Agent",
    page_icon="🔗",
    layout="wide"
)

# Title and description
st.title("Base Blockchain AI Agent 🤖")
st.markdown("""
This agent helps you interact with the Base blockchain through natural language.
You can perform various operations like:
- Creating and managing wallets
- Checking balances
- Sending transactions
- Deploying smart contracts
- And more!
""")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What would you like to do?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Get the response from the agent
            response = handle_user_input(st.session_state.agent_executor, prompt)
            
            # Stream the response
            for partial_response in stream_response(response):
                response_placeholder.markdown(partial_response + "▌")
                full_response = partial_response
            
            # Show final response without cursor
            response_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            response_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar with examples
with st.sidebar:
    st.header("Example Commands")
    example_commands = [
        "Create a new wallet",
        "Check my wallet balance",
        "Request test funds",
        "Send 0.1 ETH to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        "Deploy an NFT contract",
        "Get transaction history"
    ]
    
    for cmd in example_commands:
        if st.button(cmd):
            # Clear the current input and insert the example
            st.chat_input(cmd)

    # Clear chat button
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun() 