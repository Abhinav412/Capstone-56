from newspaper import Article

def extract_full_text(url:str) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"[ERROR] Failed to extract article: {e}")
        return ""