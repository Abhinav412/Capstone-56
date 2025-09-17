from smolagents.tools import tool
from utils.extract_article import extract_full_text

@tool
def extract_article_content(url: str) -> str:
    """
    Extract the main content from a news article URL.

    Args:
        url (str): The URL of the article to extract content from.

    Returns:
        str: The extracted article body text.
    """
    return extract_full_text(url)
