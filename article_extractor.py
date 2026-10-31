import requests
import trafilatura
from googlenewsdecoder import gnewsdecoder

def extract_article_text(url):
    if "news.google.com/rss/articles/" in url:
            try:
                
                decoded = gnewsdecoder(url)
            

                if decoded.get("success"):
                    url = decoded.get("decoded_url", url)
                    

            except Exception as error:
                print("Google News decoding error:", error)
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20
        )

        response.raise_for_status()
        print("HTTP Status:", response.status_code)
        print("Final URL:", response.url)
        print("HTML length:", len(response.text))

        article_text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            favor_precision=False,
            favor_recall=True
        )

        if article_text:
            return article_text, response.url

        return None, response.url

    except requests.RequestException as error:
        print("Article extraction error:", error)
        return None, url