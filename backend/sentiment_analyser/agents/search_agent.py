from tools.tavily_search import search_news

def run_search_agent(query: str):
    return search_news(query)
