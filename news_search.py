
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from bs4 import BeautifulSoup


def search_news(query, limit=5):
    """Search Google News RSS for related news coverage."""

    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        results = []

        for item in root.findall(".//item")[:limit]:
            source_element = item.find("source")

            description_html = item.findtext(
                "description", ""
            )
            
            soup = BeautifulSoup(description_html, "html.parser")

            description = soup.get_text(" ", strip=True)

            # Remove publisher name if it appears at the end
            if source_element is not None and source_element.text:
                publisher = source_element.text.strip()
                description = description.replace(publisher, "").strip()

            results.append({
                "title": item.findtext("title", "Untitled"),
                "url": item.findtext("link", ""),
                "source": (
                    source_element.text
                    if source_element is not None
                    and source_element.text
                    else "Unknown source"
                ),
                "source_url": (
                    source_element.get("url", "")
                    if source_element is not None
                    else ""
                ),
                "published": item.findtext(
                    "pubDate", "Unknown date"
                ),
                "description": description
            })

        return results

    except (requests.RequestException, ET.ParseError) as error:
        print("News search error:", error)
        return None
