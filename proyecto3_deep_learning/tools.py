from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import requests
import datetime
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

import os
from dotenv import load_dotenv

load_dotenv()
BANXICO_TOKEN = os.getenv("BANXICO_TOKEN")
HF_LOGIN_KEY = os.getenv("HF_LOGIN_KEY")
if HF_LOGIN_KEY:
    from huggingface_hub import login
    login(HF_LOGIN_KEY)

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
    
CETES_SERIES = {
    28:  "SF43936",
    91:  "SF43939",
    182: "SF43942",
    364: "SF43945",
    728: "SF349785",
}

@tool("get_cetes_rate")
def get_cetes_rate(term_days: int, date: str | None = None) -> str:
    #valid days are the ones displayed above
    if term_days not in CETES_SERIES:
        valid = ", ".join(str(k) for k in CETES_SERIES)
        return f"Invalid term '{term_days}'. Valid options are: {valid}."

    series_id = CETES_SERIES[term_days]
    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series_id}/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(url, headers=headers)
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

        fecha = datetime.datetime.strptime(obs["fecha"], "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The CETES {term_days}-day rate ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching CETES {term_days}-day rate: {exc}"


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
    
TIIE_SERIES = {
    28:  "SF43783",
    91:  "SF43878",
    182: "SF111916",
}

@tool("get_tiie_rate")
def get_tiie_rate(term_days: int, date: str | None = None) -> str:
    """
    Fetches the TIIE (Tasa de Interés Interbancaria de Equilibrio) rate
    for a given term in days from Banxico.
    Valid terms are: 28, 91, 182.
    """
    if term_days not in TIIE_SERIES:
        valid = ", ".join(str(k) for k in TIIE_SERIES)
        return f"Invalid term '{term_days}'. Valid options are: {valid}."

    series_id = TIIE_SERIES[term_days]
    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series_id}/datos"
    headers = {
        "Bmx-Token": BANXICO_TOKEN,
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(url, headers=headers)
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

        fecha = datetime.datetime.strptime(obs["fecha"], "%d/%m/%Y").strftime("%Y-%m-%d")
        value = float(obs["dato"])
        label = f"nearest to {date}" if date else "most recent"
        return f"The TIIE {term_days}-day rate ({label}) is {value:.4f}% as of {fecha}."

    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching TIIE {term_days}-day rate: {exc}"

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
    
def _get_news(ticker: str) -> list[dict]:
    t = yf.Ticker(ticker)
    news = t.news
    formated_news = []

    for i in range(len(news)):
        item = news[i]['content']
        formated_news.append({
            "pub_date": item.get("pubDate", ""),
            "content_type": item.get("contentType", ""),
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "provider": item.get("provider", {}).get("displayName", "N/A"),
        })
    return formated_news

_finbert_tokenizer = None
_finbert_model = None

def _load_finbert():
    global _finbert_tokenizer, _finbert_model
    if _finbert_model is None:
        _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _finbert_model.eval()
    return _finbert_tokenizer, _finbert_model

_LABEL_TO_SCORE = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

_SCORE_TO_LABEL = {
    lambda s: s > 0.15: "positive",
    lambda s: s < -0.15: "negative",    
}

def _bucket_label(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"

def _recency_weights(pub_dates: list[str]) -> list[float]:
    decay = 0.01  
    parsed = []
    for d in pub_dates:
        try:
            dt = datetime.datetime.fromisoformat(d.replace("Z", "+00:00"))
            parsed.append(dt)
        except (ValueError, AttributeError):
            parsed.append(None)

    valid = [dt for dt in parsed if dt is not None]
    if not valid:
        return [1.0] * len(pub_dates)

    most_recent = max(valid)
    weights = []
    for dt in parsed:
        if dt is None:
            weights.append(0.5)
        else:
            hours_old = (most_recent - dt).total_seconds() / 3600
            weights.append(float(np.exp(-decay * hours_old)))
    return weights

def _score_texts(texts: list[str]) -> list[dict]:
    """Returns a list of {label, confidence} dicts, one per input text."""
    tokenizer, model = _load_finbert()
    results = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            logits = model(**inputs).logits
            probs  = F.softmax(logits, dim=-1).squeeze()
            # FinBERT label order: positive=0, negative=1, neutral=2
            label_map = {0: "positive", 1: "negative", 2: "neutral"}
            pred_idx   = int(probs.argmax())
            results.append({
                "label":      label_map[pred_idx],
                "confidence": float(probs[pred_idx]),
            })
    return results

@tool("get_news_sentiment")
def get_news_sentiment(ticker: str) -> str:
    """Fetches recent news for a ticker and returns a FinBERT-based
    sentiment score aggregated across all available headlines."""
    articles = _get_news(ticker)
    comp_name = yf.Ticker(ticker).info.get("longName", ticker)
    if not articles:
        return f"No recent news found for {ticker}."

    texts   = [f"{a['title']}. {a['summary']}".strip() for a in articles]
    scores  = _score_texts(texts)
    weights = _recency_weights([a["pub_date"] for a in articles])

    weighted_sum   = 0.0
    total_weight   = 0.0
    scored_articles = []

    for article, score, weight in zip(articles, scores, weights):
        numeric = _LABEL_TO_SCORE[score["label"]]
        contribution = numeric * score["confidence"] * weight
        weighted_sum  += contribution
        total_weight  += weight
        scored_articles.append({
            "title":       article["title"],
            "provider":    article["provider"],
            "label":       score["label"],
            "confidence":  score["confidence"],
            "weight":      round(weight, 4),
            "pub_date":    article["pub_date"],
        })

    composite = weighted_sum / total_weight if total_weight > 0 else 0.0
    composite = max(-1.0, min(1.0, composite))
    label     = _bucket_label(composite)
    label     = label.upper()

    scored_articles.sort(
        key=lambda x: abs(_LABEL_TO_SCORE[x["label"]] * x["confidence"] * x["weight"]),
        reverse=True,
    )
    top_headlines = " --- ".join(
        f"[{a['label'].upper()} {a['confidence']:.0%}] {a['title']} ({a['provider']})"
        for a in scored_articles[:5]
    )

    return (
        f"Sentiment analysis for {comp_name} ({ticker.upper()}) across {len(articles)} recent articles: "
        f"Composite score: {composite:+.4f} ({label}). "
        f"Top influencing headlines: {top_headlines}"
    )
    
@tool("calculate_inflation_impact")
def calculate_inflation_impact(amount: float, months: int, annual_inflation_rate: float) -> str:
    monthly_rate = (1 + annual_inflation_rate / 100) ** (1 / 12) - 1
    future_equivalent = amount * (1 + monthly_rate) ** months
    purchasing_power_loss = future_equivalent - amount
    effective_value = amount - purchasing_power_loss

    return (
        f"With an annual inflation rate of {annual_inflation_rate:.2f}%, "
        f"{amount:.2f} pesos today will have the purchasing power of "
        f"{effective_value:.2f} pesos after {months} month(s). "
        f"That is a loss of {purchasing_power_loss:.2f} pesos in real value."
    )

@tool("multiply")
def multiply(a: float, b: float) -> str:
    result = a * b
    return f"The result of {a} × {b} is {result}."

@tool("respond_to_greeting")
def respond_to_greeting() -> str:
    return "Hello! I'm a financial data agent. How can I assist you today?"

@tool("respond_no_available_tool")
def respond_no_available_tool(tool_name: str) -> str:
    return f"Sorry, currently i'm capable of doing that. Check the list of avaiable tools for more information."

