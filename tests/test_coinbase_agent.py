import pytest
from unittest.mock import Mock, patch
from coinbase_agent import initialize_agent, handle_user_input, AgentState
from langchain_core.messages import HumanMessage, AIMessage

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("CDP_API_KEY_NAME", "test_cdp_key_name")
    monkeypatch.setenv("CDP_API_KEY_PRIVATE_KEY", "test_cdp_private_key")
    monkeypatch.setenv("NETWORK_ID", "base-sepolia")

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    return Mock()

def test_handle_user_input_success(mock_agent):
    """Test successful handling of user input."""
    test_response = "Test response"
    mock_agent.stream.return_value = [
        {"agent": {"messages": [AIMessage(content=test_response)]}}
    ]
    
    response = handle_user_input(mock_agent, "test message")
    assert response == test_response

def test_handle_user_input_error(mock_agent):
    """Test error handling in user input processing."""
    mock_agent.stream.side_effect = Exception("Test error")
    
    with pytest.raises(Exception) as exc_info:
        handle_user_input(mock_agent, "test message")
    assert str(exc_info.value) == "Test error"

@pytest.mark.asyncio
async def test_agent_state():
    """Test AgentState functionality."""
    state = AgentState(
        messages=[HumanMessage(content="test")],
        next_action=None,
        tool_calls=None,
        iterations=0
    )
    assert len(state["messages"]) == 1
    assert state["next_action"] is None
    assert state["tool_calls"] is None
    assert state["iterations"] == 0

@patch("coinbase_agent.ChatOpenAI")
@patch("coinbase_agent.CdpToolkit")
def test_initialize_agent(mock_cdp_toolkit, mock_chat_openai, mock_env_vars):
    """Test agent initialization."""
    mock_chat_openai.return_value = Mock()
    mock_cdp_toolkit.from_cdp_agentkit_wrapper.return_value = Mock()
    mock_cdp_toolkit.from_cdp_agentkit_wrapper.return_value.get_tools.return_value = []
    
    agent_executor, config, tools = initialize_agent()
    
    assert agent_executor is not None
    assert isinstance(config, dict)
    assert isinstance(tools, list) 