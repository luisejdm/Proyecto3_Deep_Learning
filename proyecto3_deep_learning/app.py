from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import gradio as gr

from agent import Agent
from config import AppConfig, load_environment
from model import ModelRunner
from prompts import DEFAULT_SYSTEM_PROMPT
from tools import build_default_tool_registry


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    api_key = load_environment()
    runner = ModelRunner.load(
        model_name=AppConfig.model_name,
        api_key=api_key,
        base_url=AppConfig.base_url,
    )
    tools = build_default_tool_registry()
    return Agent(runner=runner, tools=tools, system_prompt=DEFAULT_SYSTEM_PROMPT)


def run_agent(task: str, max_steps: int, temperature: float) -> tuple[str, str]:
    agent = get_agent()
    result, trace = agent.run(task, max_steps=int(max_steps), temperature=temperature, verbose=False)
    response = "" if result is None else str(result)
    trace_text = json.dumps(trace, indent=2, ensure_ascii=False)
    return response, trace_text


def build_demo() -> gr.Blocks:
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');

    /* ========== BASE ========== */
    *, *::before, *::after {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif !important;
        box-sizing: border-box !important;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Base(), title="Financial Agent", css=custom_css) as demo:
        gr.Markdown(
            """
            # Financial Agent
            Uses **LLaMA 3.3 70B** via NVIDIA NIM (OpenAI-compatible API).
            Ask for stock prices, company profiles, and more! The agent can use tools to fetch real-time data and provide accurate responses.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                task_input = gr.Textbox(
                    label="User task",
                    value="",
                    lines=3,
                    placeholder="Ask for a stock price, e.g. Tesla, Apple, Nvidia...",
                )
                with gr.Row():
                    max_steps_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Max iterations",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="Temperature",
                    )
                run_button = gr.Button("Run agent", variant="primary")

            with gr.Column(scale=1):
                response_output = gr.Textbox(label="Final response", lines=3)
                trace_output = gr.Code(label="Tool trace", language="json", elem_id="trace_output")

        run_button.click(
            fn=run_agent,
            inputs=[task_input, max_steps_slider, temperature_slider],
            outputs=[response_output, trace_output],
        )

        gr.Markdown(
            """
            # Agent User Guide

            | If you want to... | Example prompt |
            |------------------|----------------|
            | Know the latest price of a stock | "What's the current price of Apple?" |
            | Get company information | "Tell me about Tesla's business sector and industry" |
            | Build a low-risk portfolio | "Create a minimum variance portfolio with Microsoft, Google, and Amazon" |
            | Maximize return for the risk taken | "Give me the best risk-return portfolio using Apple, Nvidia, and Meta" |
            | Reduce downside risk compared to the S&P500 | "Build a portfolio that minimizes losses relative to the S&P500 using these 5 stocks" |
            | Find economic data from the US (FRED) | "What was the unemployment rate in December 2024?" |
            | Check Mexican CETES rates | "What's the 28-day CETES rate today?" |
            | Know monthly inflation in Mexico | "What was Mexico's monthly inflation last month?" |
            | Know annual inflation in Mexico | "What's the current annual inflation rate in Mexico?" |
            | Get the UDI value in Mexico | "What is the UDI value today?" |
            | Check Mexican TIE interest rates | "Show me the 91-day TIE rate" |
            | Know Mexico's central bank interest rate | "What is Mexico's target interest rate right now?" |
            | Get cross-currency exchange rates | "What's the current exchange rate for EUR/USD?" |
            | Analyze news sentiment for a stock | "What is the market sentiment around Tesla right now?" |
            | Calculate the impact of inflation | "how much would inflation in mexico affect my 1000 pesos over 5 months?" |
            | Make a fundamental analysis of a stock | "What is the fundamental analysis of Microsoft?" |
            """
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    get_agent()
    demo.launch()