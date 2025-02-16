import gradio as gr
from dotenv import load_dotenv
from coinbase_agent import handle_user_input, initialize_agent, TRANSACTION_THRESHOLD
import logging
import time
import queue
import threading
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    """Process user input through the agent and return the response."""
    try:
        logger.info(f"Starting to process message: {message}")
        
        # Single response accumulator
        full_response = []
        
        # Stream response from agent
        logger.info("Starting to stream response from agent")
        for chunk in handle_user_input(AGENT_EXECUTOR, message):
            if not chunk:
                logger.debug("Received empty chunk, skipping")
                continue
                
            logger.info(f"Processing chunk: {chunk}")
            
            # Handle transaction approval specially
            if "TRANSACTION APPROVAL REQUIRED" in chunk:
                logger.info("Transaction approval required, returning approval message")
                return (
                    "🚨 High-Value Transaction Detected 🚨\n"
                    f"A transaction exceeding ${TRANSACTION_THRESHOLD:,.2f} requires your approval.\n"
                    "Please use the 'Approve' or 'Reject' buttons below."
                )
            
            # Add new chunk to response
            full_response.append(chunk)
            
        # Return final combined response
        final_response = "\n".join(full_response) if full_response else "I apologize, but I wasn't able to process your request. Please try again."
        logger.info(f"Returning final response: {final_response[:100]}...")  # Log first 100 chars
        return final_response
            
    except Exception as e:
        logger.error(f"Error in chat_with_agent: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def respond(message, history):
    """Combined function to handle both user input and bot response"""
    if not message.strip():
        return history
        
    try:
        # Get complete response first
        response = chat_with_agent(message, history)
        
        # Update history once with complete response
        return history + [(message, response)]
            
    except Exception as e:
        logger.error(f"Error in response: {e}", exc_info=True)
        return history + [(message, f"Error: {str(e)}")]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Base Blockchain AI Agent")
    gr.Markdown("""Chat with the Base Blockchain AI Agent to manage your account and perform transactions.
    
    **Note:** Transactions exceeding $1,000 require explicit approval.""")
    
    chatbot = gr.Chatbot(
        label="Chat History",
        height=500,
        show_copy_button=True,
        bubble_full_width=False,
        render_markdown=True
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Message",
            placeholder="Type your message here...",
            show_label=False,
            container=True,
            scale=8,
            autofocus=True
        )
        submit = gr.Button("Send", variant="primary", scale=1)
    
    with gr.Row():
        clear = gr.Button("Clear Chat")
        
    gr.Examples(
        examples=[
            "Create a new wallet",
            "Check my balance",
            "Request test funds",
            "Send 1500 USDC to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        ],
        inputs=msg
    )
    
    with gr.Row():
        approve_btn = gr.Button("Approve Transaction", variant="primary")
        reject_btn = gr.Button("Reject Transaction", variant="secondary")
    
    transaction_status = gr.Textbox(label="Transaction Status", interactive=False)
    
    # Handle message submission (both Enter key and Send button)
    msg.submit(respond, [msg, chatbot], [chatbot], queue=True)
    submit.click(respond, [msg, chatbot], [chatbot], queue=True)
    
    # Clear message box after sending
    msg.submit(lambda: "", None, [msg], queue=False)
    submit.click(lambda: "", None, [msg], queue=False)
    
    # Handle clear button
    clear.click(lambda: [], None, chatbot, queue=False)
    
    # Handle approval buttons
    approve_btn.click(
        fn=handle_transaction_approval,
        inputs=[gr.Textbox(value="approve", visible=False)],
        outputs=transaction_status,
        queue=True
    )
    reject_btn.click(
        fn=handle_transaction_approval,
        inputs=[gr.Textbox(value="reject", visible=False)],
        outputs=transaction_status,
        queue=True
    )

if __name__ == "__main__":
    try:
        logger.info("Starting Gradio interface...")
        demo.queue()
        logger.info("Queue initialized")
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_api=False,
            share=False,
            max_threads=5
        )
        logger.info("Gradio interface launched successfully")
    except Exception as e:
        logger.error(f"Failed to start Gradio interface: {str(e)}", exc_info=True)
        raise 