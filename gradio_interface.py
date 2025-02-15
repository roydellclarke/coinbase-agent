import gradio as gr
from dotenv import load_dotenv
from coinbase_agent import AGENT_EXECUTOR, handle_user_input
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chat_with_agent(message, history):
    """
    Chat function for Gradio interface that uses the existing agent.
    """
    try:
        logger.info(f"Processing message: {message}")
        # Process the user input through the agent
        response = handle_user_input(AGENT_EXECUTOR, message)
        logger.info(f"Got response: {response}")
        
        # Stream the response character by character
        for i in range(len(response)):
            yield response[:i+1]
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        yield f"Error: {str(e)}"

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="Coinbase Agent Chat Interface",
    description="Chat with the Coinbase agent to manage your account and perform transactions.\n\n" +
    "Example commands:\n" +
    "• Create a new wallet\n" +
    "• Check my balance\n" +
    "• Send funds to 0x123...\n" +
    "• Get transaction history\n" +
    "• Request test funds\n" +
    "• Deploy a smart contract",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(
        share=False,
        show_api=False
    ) 