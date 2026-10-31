
import requests
from bs4 import BeautifulSoup


def search_evidence(query, limit=5):
    """Search Wikipedia for potentially relevant background evidence."""

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit
    }

    headers = {
        "User-Agent": "FakeNewsDetectorStudentProject/1.0"
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        results = []

        for item in data.get("query", {}).get("search", []):
            # Extract readable text from the returned search snippet.
            snippet = BeautifulSoup(
                item.get("snippet", ""),
                "html.parser"
            ).get_text(" ", strip=True)

            page_title = item["title"]
            page_url = (
                "https://en.wikipedia.org/wiki/"
                + requests.utils.quote(page_title.replace(" ", "_"))
            )

            results.append({
                "title": page_title,
                "snippet": snippet,
                "url": page_url
            })

        return results

    except requests.RequestException as error:
        print("Evidence search error:", error)
        return None

    except (ValueError, KeyError) as error:
        print("Could not read search results:", error)
        return None
