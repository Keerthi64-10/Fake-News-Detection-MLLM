from newspaper import Article, Config
from datetime import datetime

config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
config.request_timeout = 10

def scrape_article(url: str):
    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        # Only do nlp() if needed (this was causing punkt_tab error)
        try:
            article.nlp()
            summary = article.summary
            keywords = list(article.keywords)
        except:
            summary = ""
            keywords = []
        
        return {
            "url": url,
            "title": article.title,
            "text": article.text.strip(),
            "summary": summary,
            "keywords": keywords,
            "authors": article.authors,
            "publish_date": str(article.publish_date) if article.publish_date else None,
            "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        return {"url": url, "error": str(e)}

# Test with real URLs
if __name__ == "__main__":
    test_urls = [
        "https://www.thehindu.com/news/national/",
        # "https://www.bbc.com/news/world-asia-india",
        "https://timesofindia.indiatimes.com/india/"
    ]
    
    for url in test_urls:
        print(f"\nScraping: {url}")
        result = scrape_article(url)
        print(result)