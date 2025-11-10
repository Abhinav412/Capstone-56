import os
from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    SerperDevTool,
    EXASearchTool,
    ScrapeWebsiteTool,
    StagehandTool,
    ScrapeElementFromWebsiteTool,
    #FirecrawlScrapeWebsiteTool
)
from daily_market_intelligence_monitor.tools.finbert_local_api import FinBERTLocalApiTool
from daily_market_intelligence_monitor.tools.google_trends_financial_tool import GoogleTrendsFinancialTool

@CrewBase
class DailyMarketIntelligenceMonitorCrew:
    """DailyMarketIntelligenceMonitor crew"""

    def __init__(self):
        """Initialize the crew with Ollama Phi3 configuration"""
        # Configure Ollama LLM
        self.ollama_llm = LLM(
            model="ollama/phi3:mini",  # Use phi3 model
            base_url="http://localhost:11434",  # Default Ollama port
            temperature=0.7,
        )
        
        # Alternative: Use Phi3 medium or mini variants
        # self.ollama_llm = LLM(model="ollama/phi3:medium")
        # self.ollama_llm = LLM(model="ollama/phi3:mini")
    
    @agent
    def trends_monitor(self) -> Agent:
        return Agent(
            config=self.agents_config["trends_monitor"],
            tools=[
                SerperDevTool(),
                GoogleTrendsFinancialTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=self.ollama_llm,  # Use Ollama instead of GPT-4o-mini
        )
    
    @agent
    def news_search_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["news_search_specialist"],
            tools=[
                EXASearchTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=self.ollama_llm,
        )
    
    @agent
    def web_content_scraper(self) -> Agent:
        return Agent(
            config=self.agents_config["web_content_scraper"],
            tools=[
                ScrapeWebsiteTool(),
                #StagehandTool(),
                ScrapeElementFromWebsiteTool(),
                #FirecrawlScrapeWebsiteTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=self.ollama_llm,
        )
    
    @agent
    def finbert_sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["finbert_sentiment_analyst"],
            tools=[
                FinBERTLocalApiTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=self.ollama_llm,
        )
    
    @agent
    def financial_data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_data_analyst"],
            tools=[
                FinBERTLocalApiTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=self.ollama_llm,
        )
    
    @task
    def monitor_trending_topics(self) -> Task:
        return Task(
            config=self.tasks_config["monitor_trending_topics"],
            markdown=False,
        )
    
    @task
    def search_news_articles(self) -> Task:
        return Task(
            config=self.tasks_config["search_news_articles"],
            markdown=False,
        )
    
    @task
    def scrape_article_content(self) -> Task:
        return Task(
            config=self.tasks_config["scrape_article_content"],
            markdown=False,
        )
    
    @task
    def finbert_sentiment_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["finbert_sentiment_analysis"],
            markdown=False,
        )
    
    @task
    def comprehensive_analysis_report(self) -> Task:
        return Task(
            config=self.tasks_config["comprehensive_analysis_report"],
            markdown=False,
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the DailyMarketIntelligenceMonitor crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
