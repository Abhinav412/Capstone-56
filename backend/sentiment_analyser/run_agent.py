from agent.retrieval_agent import NewsRetrievalAgent
import config

if __name__ == "__main__":
    agent = NewsRetrievalAgent(query="NSE", interval=600, limit=5)
    agent.run_once