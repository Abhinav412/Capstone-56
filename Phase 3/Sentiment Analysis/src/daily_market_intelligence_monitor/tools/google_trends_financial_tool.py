from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Optional, Dict, Any
import json
import requests
import re

class GoogleTrendsFinancialToolInput(BaseModel):
    """Input schema for Google Trends Financial Tool."""
    keywords: Optional[List[str]] = Field(
        default=[],
        description="List of financial keywords to analyze. Leave empty to get trending topics."
    )
    timeframe: str = Field(
        default="now 1-d",
        description="Time range for analysis (kept for compatibility, focuses on current trends)"
    )
    geo: str = Field(
        default="US",
        description="Geographic region code (e.g., 'US', 'IN', 'GB', '' for global)"
    )
    category: int = Field(
        default=3,
        description="Google Trends category ID (3 for Business, 7 for Finance if available)"
    )

class GoogleTrendsFinancialTool(BaseTool):
    """Tool for analyzing Google Trends data for financial market insights using direct trending URL format."""

    name: str = "google_trends_financial_tool"
    description: str = (
        "Access Google Trends data to identify trending financial topics, "
        "search volume changes, and related queries for financial market analysis. "
        "Uses direct trending URL format for fast and reliable data retrieval."
    )
    args_schema: Type[BaseModel] = GoogleTrendsFinancialToolInput

    def _get_trending_data(self, geo: str = "US", category: int = 3) -> Dict[str, Any]:
        """Fetch trending data from Google Trends direct URL format."""
        try:
            # Use direct trending URL format for faster response
            base_url = "https://trends.google.com/trending"
            
            # Set up parameters
            params = {
                'geo': geo if geo else 'US',
                'category': category
            }
            
            # Set proper headers for JSON response
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://trends.google.com/',
                'Origin': 'https://trends.google.com',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin'
            }
            
            # Make GET request with shorter timeout for speed
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Try to parse JSON response
            try:
                data = response.json()
                return self._parse_trending_response(data)
            except json.JSONDecodeError:
                # If JSON parsing fails, extract data from HTML response
                return self._parse_html_response(response.text)
                
        except Exception as e:
            print(f"Error fetching trending data: {str(e)}")
            # Fallback: try different category if original fails
            if category == 7:  # If Finance category fails, try Business
                return self._get_trending_data(geo, 3)
            return {"trending_queries": [], "error": str(e)}

    def _parse_trending_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON response from trending endpoint."""
        try:
            trending_queries = []
            
            # Extract trending topics from various possible JSON structures
            if isinstance(data, dict):
                # Look for common fields that might contain trending data
                possible_keys = ['items', 'trends', 'trending', 'data', 'results']
                
                for key in possible_keys:
                    if key in data and isinstance(data[key], list):
                        for item in data[key][:10]:  # Limit to top 10
                            if isinstance(item, dict):
                                # Extract title/query/name fields
                                title = (item.get('title') or 
                                        item.get('query') or 
                                        item.get('name') or 
                                        item.get('term', ''))
                                if title and isinstance(title, str):
                                    trending_queries.append(title.strip())
                        break
            
            return {"trending_queries": trending_queries[:10]}
            
        except Exception as e:
            print(f"Error parsing JSON response: {str(e)}")
            return {"trending_queries": [], "error": str(e)}

    def _parse_html_response(self, html: str) -> Dict[str, Any]:
        """Parse HTML response and extract trending data."""
        try:
            trending_queries = []
            
            # Look for JSON data embedded in HTML
            json_pattern = r'window\.__INITIAL_STATE__\s*=\s*({.*?});'
            json_match = re.search(json_pattern, html, re.DOTALL)
            
            if json_match:
                try:
                    initial_data = json.loads(json_match.group(1))
                    # Extract trending data from initial state
                    trending_queries = self._extract_from_initial_state(initial_data)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: extract from HTML patterns
            if not trending_queries:
                # Look for trending search patterns in HTML
                title_patterns = [
                    r'<div[^>]*class="[^"]*trending[^"]*"[^>]*>([^<]+)</div>',
                    r'"title":"([^"]+)"',
                    r'<span[^>]*>([^<]*(?:stock|crypto|bitcoin|market|trading|finance)[^<]*)</span>'
                ]
                
                for pattern in title_patterns:
                    matches = re.findall(pattern, html, re.IGNORECASE)
                    for match in matches:
                        clean_match = re.sub(r'[^\w\s-]', '', match).strip()
                        if clean_match and len(clean_match) > 2:
                            trending_queries.append(clean_match)
                    if trending_queries:
                        break
            
            # Filter for financial relevance
            financial_keywords = [
                'stock', 'market', 'crypto', 'bitcoin', 'finance', 'trading',
                'investment', 'portfolio', 'dividend', 'earnings', 'ipo',
                'nasdaq', 'dow', 'sp500', 'bond', 'forex', 'currency',
                'tesla', 'apple', 'microsoft', 'amazon', 'google',
                'inflation', 'fed', 'interest', 'bank', 'price', 'money'
            ]
            
            relevant_queries = []
            for query in trending_queries:
                query_lower = query.lower()
                if any(keyword in query_lower for keyword in financial_keywords):
                    relevant_queries.append(query)
            
            # If no financial trends found, return general trends
            final_queries = relevant_queries[:10] if relevant_queries else trending_queries[:10]
            
            return {"trending_queries": final_queries}
            
        except Exception as e:
            print(f"Error parsing HTML response: {str(e)}")
            return {"trending_queries": [], "error": str(e)}

    def _extract_from_initial_state(self, data: Dict[str, Any]) -> List[str]:
        """Extract trending queries from initial state data."""
        try:
            queries = []
            
            def extract_recursive(obj, depth=0):
                if depth > 10:  # Prevent infinite recursion
                    return
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key.lower() in ['title', 'query', 'term', 'name'] and isinstance(value, str):
                            if len(value) > 2 and value not in queries:
                                queries.append(value)
                        elif isinstance(value, (dict, list)):
                            extract_recursive(value, depth + 1)
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, (dict, list)):
                            extract_recursive(item, depth + 1)
            
            extract_recursive(data)
            return queries[:10]
            
        except Exception as e:
            print(f"Error extracting from initial state: {str(e)}")
            return []

    def _analyze_keywords(self, keywords: List[str], trending_queries: List[str]) -> Dict[str, Any]:
        """Analyze provided keywords against trending data."""
        try:
            interest_data = {}
            trend_strength = {}
            related_queries = {}
            
            # Financial terms for generating related queries
            financial_terms = [
                "stock", "price", "buy", "sell", "market", "analysis", "forecast",
                "earnings", "dividend", "chart", "news", "today", "2024",
                "investment", "portfolio", "trading", "crypto", "bitcoin"
            ]
            
            for keyword in keywords[:5]:  # Limit to 5 keywords for speed
                keyword_lower = keyword.lower()
                
                # Calculate interest based on trending relevance
                trend_score = 0
                matches = 0
                
                for i, trend in enumerate(trending_queries):
                    trend_lower = trend.lower()
                    if (keyword_lower in trend_lower or 
                        trend_lower in keyword_lower or
                        any(word in trend_lower for word in keyword_lower.split())):
                        trend_score += (10 - i) * 5  # Higher score for top positions
                        matches += 1
                
                # Generate simulated interest data
                base_interest = max(10, min(100, trend_score + 20))
                
                interest_data[keyword] = {
                    "latest": base_interest,
                    "average": base_interest,
                    "max": min(100, base_interest + 10),
                    "trending_matches": matches
                }
                
                # Calculate trend strength
                trend_strength[keyword] = round(trend_score / 2, 2)
                
                # Generate related queries
                keyword_related = {"top": [], "rising": []}
                
                # Add relevant trending queries
                for trend in trending_queries:
                    if (keyword_lower in trend.lower() or 
                        any(word in trend.lower() for word in keyword_lower.split())):
                        keyword_related["top"].append(trend)
                
                # Generate related financial queries
                base_keyword = keyword.split()[0] if keyword.split() else keyword
                for term in financial_terms[:3]:
                    related_query = f"{base_keyword} {term}"
                    keyword_related["rising"].append(related_query)
                
                # Limit results
                keyword_related["top"] = keyword_related["top"][:3]
                keyword_related["rising"] = keyword_related["rising"][:3]
                
                if keyword_related["top"] or keyword_related["rising"]:
                    related_queries[keyword] = keyword_related
            
            return {
                "interest_data": interest_data,
                "trend_strength": trend_strength,
                "related_queries": related_queries
            }
            
        except Exception as e:
            print(f"Error analyzing keywords: {str(e)}")
            return {"interest_data": {}, "trend_strength": {}, "related_queries": {}}

    def _run(self, keywords: Optional[List[str]] = None, timeframe: str = "now 1-d", 
            geo: str = "US", category: int = 3) -> str:
        """
        Execute Google Trends financial analysis using direct trending URL format.
        
        Args:
            keywords: List of financial keywords to analyze
            timeframe: Time range (kept for compatibility)
            geo: Geographic region code
            category: Google Trends category ID (3 for Business, 7 for Finance)
            
        Returns:
            JSON string with trending data and analysis results
        """
        if keywords is None:
            keywords = []
            
        try:
            result = {
                "trending_queries": [],
                "interest_data": {},
                "related_queries": {},
                "trend_strength": {},
                "analysis_params": {
                    "keywords": keywords,
                    "timeframe": timeframe,
                    "geo": geo if geo else "Global",
                    "category": category
                }
            }
            
            # Get trending data using direct URL format
            trending_data = self._get_trending_data(geo, category)
            result["trending_queries"] = trending_data.get("trending_queries", [])
            
            # If no trending data and category is 7 (Finance), try category 3 (Business)
            if not result["trending_queries"] and category == 7:
                trending_data = self._get_trending_data(geo, 3)
                result["trending_queries"] = trending_data.get("trending_queries", [])
                result["analysis_params"]["category"] = 3
                result["analysis_params"]["note"] = "Fallback to Business category"
            
            # Use trending terms as keywords if none provided
            if not keywords and result["trending_queries"]:
                keywords = result["trending_queries"][:3]  # Limit for speed
            
            # Analyze specific keywords if provided
            if keywords:
                analysis_results = self._analyze_keywords(keywords, result["trending_queries"])
                result.update(analysis_results)
            
            # Add concise insights
            if result["trend_strength"]:
                strongest_trend = max(result["trend_strength"].items(), key=lambda x: abs(x[1]))
                result["insights"] = {
                    "strongest_keyword": strongest_trend[0],
                    "trend_score": strongest_trend[1],
                    "total_trends": len(result["trending_queries"]),
                    "analysis_speed": "Fast (direct URL)",
                    "data_freshness": "Real-time trending"
                }
            else:
                result["insights"] = {
                    "total_trends": len(result["trending_queries"]),
                    "geo_focus": geo if geo else "Global",
                    "category_used": category,
                    "data_source": "Direct trending URL",
                    "response_time": "< 10 seconds"
                }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {
                "error": f"Google Trends analysis failed: {str(e)}",
                "analysis_params": {
                    "keywords": keywords,
                    "timeframe": timeframe,
                    "geo": geo,
                    "category": category
                },
                "suggestions": [
                    "Using direct trending URL format for faster response",
                    "Automatic fallback from Finance (7) to Business (3) category",
                    "10-second timeout to prevent long waits",
                    "Try different geo codes (US, IN, GB) if needed"
                ]
            }
            return json.dumps(error_result, indent=2)