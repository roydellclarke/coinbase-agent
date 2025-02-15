import gradio as gr
from dotenv import load_dotenv
from coinbase_agent import handle_user_input, initialize_agent
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize agent globally
try:
    AGENT_EXECUTOR, AGENT_CONFIG, CDP_TOOLS = initialize_agent()
    logger.info("Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    raise

def chat_with_agent(message, history):
    """
    Chat function for Gradio interface that uses the existing agent.
    """
    try:
        logger.info(f"Processing message: {message}")
        # Process the user input through the agent
        response = handle_user_input(AGENT_EXECUTOR, message)
        logger.info(f"Got response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="Coinbase Agent Chat Interface",
    description="""Chat with the Coinbase agent to manage your account and perform transactions.

Example commands:
• Create a new wallet
• Check my balance
• Send funds to an address
• Get transaction history
• Request test funds
• Deploy a smart contract""",
    examples=[
        "Create a new wallet",
        "Check my balance",
        "Request test funds",
        "Deploy an NFT contract"
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",  # Make accessible from other devices
        share=False,  # Set to True to create a public link
        show_api=False
    ) 