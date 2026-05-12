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
        return f"The monthly inflation rate in Mexico ({label}) is {value:.4f}% as of {fecha}."
    
    except ValueError:
        return f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g. 2024-01-15)."
    except Exception as exc:
        return f"Error fetching monthly inflation rate in Mexico: {exc}"
    
@tool("get_inflation_mexico")
def get_inflation_mexico(date: str | None = None) -> str:
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

# --------------- sector-adjusted valuation thresholds -------------------------
# Tuple layout: (pe_strong, pe_weak, pb_strong, pb_weak, ev_ebitda_strong, ev_ebitda_weak)
# "strong" means the value that earns the maximum score of 2.
# "weak"   means the value that earns the minimum score of 0.
# Values between the two thresholds score 1 (neutral).
_SECTOR_VALUATION_THRESHOLDS: dict[str, tuple] = {
    "technology":         (25, 45, 4.0, 10.0, 15, 30),
    "healthcare":         (20, 35, 3.0,  8.0, 14, 25),
    "financial-services": (12, 20, 1.0,  2.5, 10, 18),
    "consumer-cyclical":  (18, 30, 2.5,  6.0, 12, 22),
    "consumer-defensive": (18, 28, 3.0,  6.0, 12, 20),
    "energy":             (10, 20, 1.5,  3.0,  6, 14),
    "basic-materials":    (12, 22, 1.5,  3.5,  8, 16),
    "industrials":        (18, 30, 2.5,  5.0, 12, 20),
    "real-estate":        (30, 55, 1.5,  3.5, 18, 30),
    "utilities":          (15, 25, 1.5,  3.0, 10, 18),
    "default":            (18, 35, 2.5,  6.0, 12, 22),
}
 
 
def _valuation_thresholds(sector_key: str | None) -> tuple:
    key = (sector_key or "").lower()
    return _SECTOR_VALUATION_THRESHOLDS.get(key, _SECTOR_VALUATION_THRESHOLDS["default"])
 
 
def _score_metric(value: float, strong_threshold: float, weak_threshold: float,
                  lower_is_better: bool = True) -> int:
    """
    Scores a single metric on a 0–2 scale.
 
    For lower_is_better metrics (P/E, D/E, EV/EBITDA …):
        value <= strong_threshold → 2
        value >= weak_threshold   → 0
        in between                → 1
 
    For higher_is_better metrics (ROE, margins, FCF yield …):
        value >= strong_threshold → 2
        value <= weak_threshold   → 0
        in between                → 1
    """
    if lower_is_better:
        if value <= strong_threshold:
            return 2
        if value >= weak_threshold:
            return 0
        return 1
    else:
        if value >= strong_threshold:
            return 2
        if value <= weak_threshold:
            return 0
        return 1
 
 
@tool("get_fundamental_analysis")
def get_fundamental_analysis(ticker: str) -> str:
    """
    Performs a quantitative fundamental analysis scorecard for a given ticker.
 
    Evaluates 15 metrics across four categories:
      - Valuation      (P/E, P/B, EV/EBITDA, PEG)               max  8 pts
      - Profitability  (ROE, ROA, Gross margin, Net margin)       max  8 pts
      - Financial Health (D/E, Current ratio, IC, FCF yield)     max  8 pts
      - Growth         (Revenue growth, Earnings growth, Div yield) max 6 pts
                                                                 ──────────
                                                          TOTAL   max 30 pts
 
    Scoring per metric: 2 = strong, 1 = neutral / data unavailable, 0 = weak.
    Valuation thresholds are sector-adjusted via yfinance sectorKey.
    Composite: ≥70% → BUY | 40–69% → HOLD | <40% → SELL.
 
    Apply the same exchange suffix rules as get_price_on_date (e.g. BIMBOA.MX).
    """
    t    = yf.Ticker(ticker)
    info = t.info
 
    company_name = info.get("longName", ticker)
    sector_key   = info.get("sectorKey", None)
    sector_label = info.get("sector", "Unknown sector")
 
    pe_s, pe_w, pb_s, pb_w, ev_s, ev_w = _valuation_thresholds(sector_key)
 
    def safe(key: str, scale: float = 1.0):
        """Returns (scaled_value, is_available). Missing or non-numeric → (None, False)."""
        raw = info.get(key)
        if raw is None or not isinstance(raw, (int, float)):
            return None, False
        return raw * scale, True
 
    # ── Valuation (max 8 pts) ──────────────────────────────────────────────────
    pe,        pe_ok  = safe("trailingPE")
    pb,        pb_ok  = safe("priceToBook")
    ev_ebitda, ev_ok  = safe("enterpriseToEbitda")
    peg,       peg_ok = safe("pegRatio")
 
    pe_score  = _score_metric(pe,        pe_s, pe_w, lower_is_better=True) if pe_ok  else 1
    pb_score  = _score_metric(pb,        pb_s, pb_w, lower_is_better=True) if pb_ok  else 1
    ev_score  = _score_metric(ev_ebitda, ev_s, ev_w, lower_is_better=True) if ev_ok  else 1
    peg_score = _score_metric(peg,       1.0,  2.0,  lower_is_better=True) if peg_ok else 1
 
    valuation_score = pe_score + pb_score + ev_score + peg_score  # max 8
 
    # ── Profitability (max 8 pts) ──────────────────────────────────────────────
    roe,     roe_ok = safe("returnOnEquity", scale=100)
    roa,     roa_ok = safe("returnOnAssets", scale=100)
    gross_m, gm_ok  = safe("grossMargins",   scale=100)
    net_m,   nm_ok  = safe("profitMargins",  scale=100)
 
    roe_score = _score_metric(roe,     15.0, 8.0,  lower_is_better=False) if roe_ok else 1
    roa_score = _score_metric(roa,      5.0, 2.0,  lower_is_better=False) if roa_ok else 1
    gm_score  = _score_metric(gross_m, 40.0, 20.0, lower_is_better=False) if gm_ok  else 1
    nm_score  = _score_metric(net_m,   10.0,  3.0, lower_is_better=False) if nm_ok  else 1
 
    profit_score = roe_score + roa_score + gm_score + nm_score  # max 8
 
    # ── Financial Health (max 8 pts) ───────────────────────────────────────────
    de,      de_ok   = safe("debtToEquity")
    cr,      cr_ok   = safe("currentRatio")
    ebitda,  ebit_ok = safe("ebitda")
    int_exp, ie_ok   = safe("interestExpense")
    fcf,     fcf_ok  = safe("freeCashflow")
    mktcap,  mc_ok   = safe("marketCap")
 
    # yfinance returns D/E as a percentage (e.g. 150 means 1.50); normalise to ratio.
    de_adj   = de / 100.0 if de_ok else None
    de_score = _score_metric(de_adj, 0.5, 1.5, lower_is_better=True) if de_ok else 1
 
    cr_score = _score_metric(cr, 2.0, 1.0, lower_is_better=False) if cr_ok else 1
 
    # Interest coverage = EBITDA / |interest expense|; higher is better.
    if ebit_ok and ie_ok and int_exp != 0:
        ic       = abs(ebitda) / abs(int_exp)
        ic_score = _score_metric(ic, 5.0, 2.0, lower_is_better=False)
    else:
        ic       = None
        ic_score = 1
 
    # FCF yield = FCF / market cap (%); >5% strong, <0% weak.
    if fcf_ok and mc_ok and mktcap > 0:
        fcf_yield = (fcf / mktcap) * 100
        fcf_score = _score_metric(fcf_yield, 5.0, 0.0, lower_is_better=False)
    else:
        fcf_yield = None
        fcf_score = 1
 
    health_score = de_score + cr_score + ic_score + fcf_score  # max 8
 
    # ── Growth (max 6 pts) ─────────────────────────────────────────────────────
    rev_g,  rg_ok = safe("revenueGrowth",  scale=100)
    earn_g, eg_ok = safe("earningsGrowth", scale=100)
    div_y,  dy_ok = safe("dividendYield",  scale=100)
 
    rev_score  = _score_metric(rev_g,  10.0, 0.0, lower_is_better=False) if rg_ok else 1
    earn_score = _score_metric(earn_g, 10.0, 0.0, lower_is_better=False) if eg_ok else 1
 
    # Dividend yield: 2–5.5% is the ideal income range.
    # Below 0.5% is neutral (growth company, no penalty). Above 6% may signal distress.
    if dy_ok:
        if 2.0 <= div_y <= 5.5:
            div_score = 2
        elif div_y > 6.0 or div_y < 0.5:
            div_score = 0
        else:
            div_score = 1
    else:
        div_score = 1  # no dividend data → neutral
 
    growth_score = rev_score + earn_score + div_score  # max 6
 
    # ── Composite & recommendation ─────────────────────────────────────────────
    MAX_SCORE = 30
    composite = valuation_score + profit_score + health_score + growth_score
    pct       = composite / MAX_SCORE
 
    if pct >= 0.70:
        recommendation = "BUY"
        rationale      = "the company scores strongly across most fundamental dimensions"
    elif pct >= 0.40:
        recommendation = "HOLD"
        rationale      = "the fundamentals are mixed with no compelling entry or exit signal"
    else:
        recommendation = "SELL"
        rationale      = "the company shows material weakness across multiple fundamental dimensions"
 
    def fmt(value, decimals: int = 2, suffix: str = "") -> str:
        return "N/A" if value is None else f"{value:.{decimals}f}{suffix}"
 
    return (
        f"Fundamental analysis scorecard for {company_name} ({ticker.upper()}) | Sector: {sector_label}. "
        f"VALUATION ({valuation_score}/8): "
        f"P/E {fmt(pe)}x [score {pe_score}/2], "
        f"P/B {fmt(pb)}x [score {pb_score}/2], "
        f"EV/EBITDA {fmt(ev_ebitda)}x [score {ev_score}/2], "
        f"PEG {fmt(peg)} [score {peg_score}/2]. "
        f"PROFITABILITY ({profit_score}/8): "
        f"ROE {fmt(roe)}% [score {roe_score}/2], "
        f"ROA {fmt(roa)}% [score {roa_score}/2], "
        f"Gross margin {fmt(gross_m)}% [score {gm_score}/2], "
        f"Net margin {fmt(net_m)}% [score {nm_score}/2]. "
        f"FINANCIAL HEALTH ({health_score}/8): "
        f"D/E {fmt(de_adj)} [score {de_score}/2], "
        f"Current ratio {fmt(cr)} [score {cr_score}/2], "
        f"Interest coverage {fmt(ic)}x [score {ic_score}/2], "
        f"FCF yield {fmt(fcf_yield)}% [score {fcf_score}/2]. "
        f"GROWTH ({growth_score}/6): "
        f"Revenue growth {fmt(rev_g)}% [score {rev_score}/2], "
        f"Earnings growth {fmt(earn_g)}% [score {earn_score}/2], "
        f"Dividend yield {fmt(div_y)}% [score {div_score}/2]. "
        f"COMPOSITE SCORE: {composite}/{MAX_SCORE} ({pct:.0%}). "
        f"RECOMMENDATION: {recommendation} — {rationale}."
    )





@tool("respond_to_greeting")
def respond_to_greeting() -> str:
    return "Hello! I'm a financial data agent. How can I assist you today?"

@tool("respond_no_available_tool")
def respond_no_available_tool(tool_name: str) -> str:
    return f"Sorry, currently i'm capable of doing that. Check the list of avaiable tools for more information."

