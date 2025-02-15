import pytest
from unittest.mock import Mock, patch, mock_open, ANY
from coinbase_agent import initialize_agent, handle_user_input, AgentState, graph
from langchain_core.messages import HumanMessage, AIMessage
from cdp.errors import InvalidAPIKeyFormatError

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
    mock = Mock()
    mock.invoke.return_value = {
        "messages": [AIMessage(content="Test response")]
    }
    return mock

def test_handle_user_input_success(monkeypatch):
    """Test successful handling of user input."""
    # Mock the graph.invoke function
    mock_graph = Mock()
    mock_graph.invoke.return_value = {
        "messages": [AIMessage(content="Test response")]
    }
    monkeypatch.setattr("coinbase_agent.graph", mock_graph)
    
    response = handle_user_input(Mock(), "test message")
    assert response == "Test response"

def test_handle_user_input_error(monkeypatch):
    """Test error handling in user input processing."""
    # Mock the graph.invoke function to raise an exception
    mock_graph = Mock()
    mock_graph.invoke.side_effect = Exception("Test error")
    monkeypatch.setattr("coinbase_agent.graph", mock_graph)
    
    with pytest.raises(Exception) as exc_info:
        handle_user_input(Mock(), "test message")
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
@patch("coinbase_agent.CdpAgentkitWrapper")
@patch("builtins.open", new_callable=mock_open, read_data="test_wallet_data")
@patch("os.path.exists")
def test_initialize_agent(mock_exists, mock_file, mock_wrapper, mock_cdp_toolkit, mock_chat_openai, mock_env_vars):
    """Test agent initialization."""
    # Setup mocks
    mock_exists.return_value = True
    mock_chat_openai.return_value = Mock()
    mock_cdp_toolkit.from_cdp_agentkit_wrapper.return_value = Mock()
    mock_cdp_toolkit.from_cdp_agentkit_wrapper.return_value.get_tools.return_value = []
    mock_wrapper.return_value = Mock()
    mock_wrapper.return_value.export_wallet.return_value = "test_wallet_data"
    
    # Test successful initialization
    agent_executor, config, tools = initialize_agent()
    
    assert agent_executor is not None
    assert isinstance(config, dict)
    assert isinstance(tools, list)
    
    # Verify file operations
    mock_file.assert_called()
    # Verify that exists was called with wallet_data.txt at least once
    mock_exists.assert_any_call("wallet_data.txt") 