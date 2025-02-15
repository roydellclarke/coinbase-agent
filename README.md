# Coinbase Developer Platform Agent

A chatbot interface for interacting with the Coinbase Developer Platform using the CDP Agentkit.

## Features

- Interactive chat interface using Gradio
- Blockchain interactions via CDP Agentkit
- Support for wallet management, smart contracts, NFTs, and more
- Real-time streaming responses
- Environment variable configuration

## Prerequisites

- Python 3.10+
- Conda (recommended for environment management)
- Coinbase Developer Platform API Key
- OpenAI API Key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/coinbase-agent.git
cd coinbase-agent
```

2. Create and activate a Conda environment:
```bash
conda create -n coinbase python=3.10
conda activate coinbase
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY="your_openai_api_key"
CDP_API_KEY_NAME="your_cdp_api_key_name"
CDP_API_KEY_PRIVATE_KEY="your_cdp_private_key"
NETWORK_ID="base-sepolia"
```

## Usage

Run the Gradio interface:
```bash
python gradio_interface.py
```

Or run the CLI version:
```bash
python coinbase_agent.py
```

## Available Commands

- Create a new wallet
- Check wallet balance
- Send funds
- Get transaction history
- Request test funds
- Deploy smart contracts
- And more!

## Development

To run tests:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 