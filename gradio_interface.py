import gradio as gr
from dotenv import load_dotenv
from coinbase_agent import handle_user_input, initialize_agent, TRANSACTION_THRESHOLD
import logging
import time
import queue
import threading

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

# Global approval queue for handling transaction approvals
approval_queue = queue.Queue()
current_transaction = None

def handle_transaction_approval(choice: str):
    """Handle the user's approval choice"""
    global current_transaction
    if current_transaction is not None:
        approval_queue.put(choice.lower() == "approve")
        current_transaction = None
    return "Transaction decision recorded."

def chat_with_agent(message, history):
    """
    Chat function for Gradio interface that uses the existing agent.
    """
    global current_transaction
    
    try:
        logger.info(f"Processing message: {message}")
        response = handle_user_input(AGENT_EXECUTOR, message)
        response_lines = list(dict.fromkeys(response.split('\n')))
        response = '\n'.join(response_lines)
        
        # Check if this is a transaction approval request
        if "TRANSACTION APPROVAL REQUIRED" in response:
            current_transaction = response
            # Create a more user-friendly approval message
            approval_msg = (
                "🚨 High-Value Transaction Detected 🚨\n"
                f"A transaction exceeding ${TRANSACTION_THRESHOLD:,.2f} requires your approval.\n"
                "Please use the 'Approve' or 'Reject' buttons below."
            )
            return approval_msg
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Base Blockchain AI Agent")
    gr.Markdown("""Chat with the Base Blockchain AI Agent to manage your account and perform transactions.
    
    **Note:** Transactions exceeding $1,000 require explicit approval.""")
    
    chatbot = gr.ChatInterface(
        fn=chat_with_agent,
        examples=[
            "Create a new wallet",
            "Check my balance",
            "Request test funds",
            "Send 1500 USDC to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e"  # Example of high-value transaction
        ],
        title=""
    )
    
    with gr.Row():
        approve_btn = gr.Button("Approve Transaction", variant="primary")
        reject_btn = gr.Button("Reject Transaction", variant="secondary")
    
    transaction_status = gr.Textbox(label="Transaction Status", interactive=False)
    
    # Handle approval buttons
    approve_btn.click(
        fn=handle_transaction_approval,
        inputs=[gr.Textbox(value="approve", visible=False)],
        outputs=transaction_status
    )
    reject_btn.click(
        fn=handle_transaction_approval,
        inputs=[gr.Textbox(value="reject", visible=False)],
        outputs=transaction_status
    )

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False
    ) 