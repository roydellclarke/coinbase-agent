import gradio as gr
from coinbase_agent import initialize_agent, get_response
import logging

# Initialize the agent
try:
    AGENT_EXECUTOR, AGENT_CONFIG, CDP_TOOLS = initialize_agent()
    initialization_error = None
except Exception as e:
    initialization_error = str(e)
    logging.error(f"Failed to initialize agent: {e}")

def chat_with_agent(message, history):
    """Chat function for Gradio interface"""
    if initialization_error:
        return f"Error: Agent initialization failed - {initialization_error}"
    
    try:
        response = get_response(message)
        return response
    except Exception as e:
        logging.error(f"Error during chat: {e}")
        return f"Error: {str(e)}"

# Create the Gradio interface
demo = gr.ChatInterface(
    chat_with_agent,
    title="Coinbase Developer Platform Agent",
    description="""This agent can help you interact with the Coinbase Developer Platform.
    You can:
    - Create and manage wallets
    - Deploy smart contracts
    - Mint NFTs
    - Create tokens
    - And more!
    """,
    theme="soft",
    examples=[
        "Create a new wallet",
        "Deploy an NFT collection",
        "Create an ERC20 token",
        "Show my wallet balance",
    ],
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear Chat",
)

if __name__ == "__main__":
    if initialization_error:
        print(f"Warning: Agent initialization failed - {initialization_error}")
        print("The interface will start but may not function correctly.")
    
    demo.launch(
        server_name="0.0.0.0",  # Make accessible from other devices
        share=False,  # Set to True to create a public link
        show_api=False,
    ) 