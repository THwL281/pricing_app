import pandas as pd
import requests
from bs4 import BeautifulSoup

def fetch_competitor_prices_stub(sku_list: list[str]) -> pd.DataFrame:
    """
    Replace this stub with:
    - a price aggregator API, OR
    - your own internal feed, OR
    - compliant scraping (robots.txt, ToS, rate limits)
    """
    rows = []
    for sku in sku_list:
        rows.append({"sku": sku, "competitor": "ExampleCompetitor", "comp_price": None})
    return pd.DataFrame(rows)

def scrape_example_product_price(url: str) -> float | None:
    """
    Example "how" only. You MUST ensure you have permission / comply with ToS and robots.txt.
    """
    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # This selector is site-specific; you must adapt it:
    price_el = soup.select_one("[data-testid='price']")
    if not price_el:
        return None

    txt = price_el.get_text(strip=True)
    # naive parsing; adapt for locales/currency
    digits = "".join(ch for ch in txt if (ch.isdigit() or ch in ".,"))[:20]
    digits = digits.replace(",", ".")
    try:
        return float(digits)
    except:
        return None
