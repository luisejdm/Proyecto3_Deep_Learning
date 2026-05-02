from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import requests
import datetime
import pandas as pd

import os
from dotenv import load_dotenv

load_dotenv()
BANXICO_TOKEN = os.getenv("BANXICO_TOKEN")


ToolFunction = Callable[..., object]

_DEFAULT_TOOL_FUNCTIONS: dict[str, ToolFunction] = {}


def tool(name: str | None = None):
    """Decorator that registers a function in the default tool registry."""
    def decorator(function: ToolFunction) -> ToolFunction:
        _DEFAULT_TOOL_FUNCTIONS[name or function.__name__] = function
        return function
    return decorator


@dataclass
class ToolRegistry:
    tools: dict[str, ToolFunction] = field(default_factory=dict)

    def register(self, name: str, function: ToolFunction) -> None:
        self.tools[name] = function

    def execute(self, name: str, *args) -> object:
        if name not in self.tools:
            raise KeyError(f"tool '{name}' does not exist")
        return self.tools[name](*args)

    def names(self) -> list[str]:
        return list(self.tools.keys())


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name, function in _DEFAULT_TOOL_FUNCTIONS.items():
        registry.register(name, function)
    return registry


@tool("get_price_on_date")
def get_price_on_date(ticker, date=None):
    t = yf.Ticker(ticker)
    
    use_default_date = date is None
    
    if date is None:
        date = datetime.date.today()
    else:
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    
    data = pd.DataFrame(t.history(start=date - datetime.timedelta(days=5), end=date + datetime.timedelta(days=5))['Close'])    
    if data.empty:
        return f"No price data available for {t.ticker} around {date}."
    
    data['Date'] = data.index.date
    data['DateDiff'] = np.abs(data['Date'] - date)
    nearest_row = data.loc[data['DateDiff'].idxmin()]
    price = nearest_row['Close']
    actual_date = nearest_row['Date']
    official_name = t.info['longName']
    
    if use_default_date:
        return f"The last price of {official_name} ({t.ticker}) is ${price:.2f} as of {actual_date}."
    else:
        return f"The price of {official_name} ({t.ticker}) nearest to {date} was ${price:.2f} on {actual_date}."

@tool("get_company_profile")
def get_company_profile_tool(ticker: str) -> str:
    t = yf.Ticker(ticker)
    info = t.info
    official_name = info['longName']
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    description = info.get('longBusinessSummary', 'No description available.')
    return (
        f"{official_name} operates in the {sector} sector and {industry} industry. "
        f"Company profile: {description}"
    )

@tool("min_variance_portfolio")
def min_variance_portfolio(*tickers: str) -> str:
    ticker_list = list(tickers)
    data = yf.download(ticker_list, period="2y", progress=False)['Close'][ticker_list]
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()
    mean_rt = returns.mean()

    variance = lambda w: w.T @ cov_matrix @ w
    x0 = np.ones(len(ticker_list)) / len(ticker_list)
    bounds = [(0, 3)] * len(ticker_list)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    result = minimize(variance, x0, bounds=bounds, constraints=constraints, tol=1e-16, method='SLSQP')
    return (
        f"Optimal weights for minimum variance portfolio:\n"
        f"{ {ticker_list[i]: round(w, 4) for i, w in enumerate(result.x)} }\n"
        f"Expected annual return: {(mean_rt @ result.x * 252):.2%}\n"
        f"Annualized volatility: {(np.sqrt(result.x.T @ cov_matrix @ result.x) * np.sqrt(252)):.2%}"
    )

@tool("max_sharpe_portfolio")
def max_sharpe_portfolio(*tickers: str) -> str:
    ticker_list = list(tickers)
    data = yf.download(ticker_list, period="2y", progress=False)['Close'][ticker_list]
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()
    mean_rt = returns.mean()

    sharpe = lambda w: -(mean_rt @ w) / np.sqrt(w.T @ cov_matrix @ w)
    x0 = np.ones(len(ticker_list)) / len(ticker_list)
    bounds = [(0, 3)] * len(ticker_list)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    result = minimize(sharpe, x0, bounds=bounds, constraints=constraints, tol=1e-16, method='SLSQP')
    return (
        f"Optimal weights for maximum Sharpe ratio portfolio:\n"
        f"{ {ticker_list[i]: round(w, 4) for i, w in enumerate(result.x)} }\n"
        f"Expected annual return: {(mean_rt @ result.x * 252):.2%}\n"
        f"Annualized volatility: {(np.sqrt(result.x.T @ cov_matrix @ result.x) * np.sqrt(252)):.2%}"
    )

@tool("min_target_semivariance_portfolio")
def min_target_semivariance_portfolio(*tickers: str) -> str:
    ticker_list = list(tickers)
    data = yf.download(ticker_list, period="2y", progress=False)['Close'][ticker_list]
    returns = data.pct_change().dropna()
    corr = returns.corr()
    cov_matrix = returns.cov()
    benchmark = yf.download("^GSPC", period="2y", progress=False)['Close'].pct_change().dropna()
    differences = returns - benchmark.values
    below_zero_target = differences[differences < 0].fillna(0)
    target_downside = np.array(below_zero_target.std())
    target_semivariance = np.multiply(target_downside.reshape(len(target_downside), 1), target_downside) * corr

    semivariance = lambda w: w.T @ target_semivariance @ w
    x0 = np.ones(len(ticker_list)) / len(ticker_list)
    bounds = [(0, 3)] * len(ticker_list)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    result = minimize(semivariance, x0, bounds=bounds, constraints=constraints, tol=1e-16, method='SLSQP')
    return (
        f"Optimal weights for minimum target semivariance portfolio:\n"
        f"{ {ticker_list[i]: round(w, 4) for i, w in enumerate(result.x)} }\n"
        f"Expected annual return: {(returns.mean() @ result.x * 252):.2%}\n"
        f"Annualized volatility: {(np.sqrt(result.x.T @ cov_matrix @ result.x) * np.sqrt(252)):.2%}"
    )
    
@tool("get_cetes_28")
def get_cetes_28(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43936/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The CETES 28-day rate ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching CETES 28-day rate: {exc}"

@tool("get_cetes_91")
def get_cetes_91(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43939/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The CETES 91-day rate ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching CETES 91-day rate: {exc}"
    
@tool("get_cetes_182")
def get_cetes_182(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43942/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The CETES 182-day rate ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching CETES 182-day rate: {exc}"
    
@tool("get_cetes_364")
def get_cetes_364(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43945/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The CETES 364-day rate ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching CETES 364-day rate: {exc}"
    
@tool("get_cetes_728")
def get_cetes_728(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF349785/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The CETES 728-day rate ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching CETES 728-day rate: {exc}"
    
@tool("get_mensual_inflation_mexico")
def get_mensual_inflation_mexico(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SP30578/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The monthly inflation rate in Mexico ({label}) is {value:.4f}% as of {fecha}."
    
    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching monthly inflation rate in Mexico: {exc}"
    
@tool("get_inflation_mexico")
def get_inflation_mexico(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SP30577/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The annual inflation rate in Mexico ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching annual inflation rate in Mexico: {exc}"
    
@tool("get_udis")
def get_udis(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SP68257/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The value of UDIs in Mexico ({label}) is {value:.4f} MXN as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching UDIs value in Mexico: {exc}"
    
@tool("get_tie_28")
def get_tie_28(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43783/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The TIE 28-day rate ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching TIE 28-day rate: {exc}"
    
@tool("get_tie_91")
def get_tie_91(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43878/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The TIE 91-day rate ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching TIE 91-day rate: {exc}"
    
@tool("get_tie_182")
def get_tie_182(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF111916/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The TIE 182-day rate ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching TIE 182-day rate: {exc}"
    
@tool("get_target_interest_rate_mexico")
def get_target_interest_rate_mexico(date: str | None = None) -> str:
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF61745/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        obs_list = response.json()["bmx"]["series"][0]["datos"]

        if date is None:
            obs = obs_list[-1]
        else:
            target = datetime.datetime.strptime(date, "%Y-%m-%d")
            obs = min(
                obs_list,
                key=lambda o: abs(datetime.datetime.strptime(o["fecha"], "%d/%m/%Y") - target),
            )

        fecha = obs["fecha"]
        fecha = datetime.datetime.strptime(fecha, "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The target interest rate in Mexico ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching target interest rate in Mexico: {exc}"
    
@tool("get_exchange_rate")
def get_exchange_rate(base: str, quote: str, date: str | None = None) -> str:
    base = base.strip().upper()
    quote = quote.strip().upper()
    ticker_symbol = f"{base}{quote}=X"
 
    try:
        if date is None:
            target_date = datetime.date.today()
        else:
            target_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
 
        t = yf.Ticker(ticker_symbol)
        data = t.history(
            start=target_date - datetime.timedelta(days=7),
            end=target_date + datetime.timedelta(days=7),
        )
 
        if data.empty:
            return (
                f"No exchange rate data found for {base}/{quote} ({ticker_symbol}). "
                f"Verify that both currency codes are valid ISO 4217 codes."
            )
 
        data["Date"] = data.index.date
        data["DateDiff"] = data["Date"].apply(lambda d: abs((d - target_date).days))
        nearest = data.loc[data["DateDiff"].idxmin()]
        rate = nearest["Close"]
        actual_date = nearest["Date"]
        date_label = f"nearest to {date}" if date else "most recent"
        return f"The exchange rate for {base}/{quote} ({date_label}) is {rate:.6f} as of {actual_date}." 
        
    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching exchange rate for {base}/{quote}: {exc}"
    
@tool("respond_to_greeting")
def respond_to_greeting() -> str:
    return "Hello! I'm a financial data agent. How can I assist you today?"

@tool("respond_no_available_tool")
def respond_no_available_tool(tool_name: str) -> str:
    return f"Sorry, currently i'm capable of doing that. Check the list of avaiable tools for more information."
