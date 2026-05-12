# Proyecto3_Deep_Learning

## Project Organization

```
├── LICENSE # MIT License file
├── README.md # Project documentation and setup guide
├── requirements.txt # Python dependencies
│
├── notebooks/ # Jupyter notebooks for exploration and testing
│ ├── 0.01-lejdm-tool-testing.ipynb # Initial exploration and testing of financial tools
│ ├── 3.01-lejdm-trying-qwen2.5-1.5B.ipynb # Testing Qwen2.5 model for potential alternatives
│ ├── 3.02-lejdm-sentimental-finbert.ipynb # Sentiment analysis experiments with FinBERT
│ └── 3.03-smdo-details_and_optimizations.ipynb # Agent detail improvements and optimizations
│
└── proyecto3_deep_learning/ # Main agent package
  ├── agent.py # Core Agent class that orchestrates tool calling and response handling
  ├── app.py # Gradio web interface for user interaction
  ├── config.py # Configuration (model name, API keys, environment loading)
  ├── main.py # Entry point to launch the Gradio app
  ├── model.py # API-based model runner (LLaMA via NVIDIA NIM)
  ├── parsing.py # Parses LLM responses into ACTION / FINAL commands
  ├── prompts.py # System prompt with tool definitions and usage rules
  └── tools.py # Registry of all available tools (stock prices, portfolio optimization, sentiment, fundamentals, Banxico data, etc.)
```

Financial tool-using agent with a Gradio interface. The system uses an LLM through NVIDIA NIM (OpenAI-compatible API), forces tool-calling behavior with a strict prompt, executes financial data tools, and returns both final answer and execution trace.

## Overview

This project implements an interactive financial assistant focused on:

- Stock prices (current or nearest to a target date)
- Company profiles (sector, industry, business summary)
- Portfolio optimization (minimum variance, maximum Sharpe, target semivariance)
- Mexican macro indicators from Banxico (CETES, TIIE, inflation, UDIS, target rate)
- FX cross rates (e.g., EUR/USD, USD/CAD)
- News sentiment analysis powered by FinBERT (recency-weighted, per-ticker)
- Fundamental analysis (valuation metrics: P/E, P/B, EV/EBITDA, scored by sector)
- Inflation impact calculator and basic arithmetic utilities

The app is designed around an agent loop:

1. User enters a natural-language request.
2. Model answers in strict `ACTION:` / `FINAL:` format.
3. Agent executes tool calls from a registry.
4. Tool observation is fed back to the model.
5. Final answer is returned only after tool execution.

## Project Approach

- Prompt-first orchestration: the system prompt defines routing and response rules.
- Deterministic tool interface: tools are registered with a decorator and called by name.
- Runtime safety: parser validates model output format before execution.
- Transparent behavior: UI shows final response and JSON tool trace side by side.

## Primary Dependencies

Only direct, code-level dependencies are used:

- `gradio`
- `openai`
- `python-dotenv`
- `yfinance`
- `numpy`
- `pandas`
- `scipy`
- `requests`
- `transformers`
- `torch`

## Environment Variables

Create a `.env` file in the project root with:

```env
NVIDIA_API_KEY=your_nvidia_nim_api_key
BANXICO_TOKEN=your_banxico_token
```

## Run

```bash
python -m pip install -r requirements.txt
python proyecto3_deep_learning/main.py
```