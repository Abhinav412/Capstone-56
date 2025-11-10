from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List, Union
import requests
import json

class FinBERTAnalysisInput(BaseModel):
    """Input schema for FinBERT Local API Tool."""
    text: Union[str, List[str]] = Field(
        ..., 
        description="Text to analyze for sentiment. Can be a single string or list of strings for batch analysis."
    )

class FinBERTLocalApiTool(BaseTool):
    """Tool for analyzing financial text sentiment using a local FinBERT API."""

    name: str = "finbert_local_api"
    description: str = (
        "Analyzes sentiment of financial text using a local FinBERT API. "
        "Returns sentiment label (positive/negative/neutral), confidence score, "
        "and detailed probability scores for each sentiment class. "
        "Supports both single text and batch analysis."
    )
    args_schema: Type[BaseModel] = FinBERTAnalysisInput

    def _run(self, text: Union[str, List[str]]) -> str:
        """
        Analyze sentiment using local FinBERT API.
        
        Args:
            text: Text to analyze (string or list of strings)
            
        Returns:
            JSON string with sentiment analysis results
        """
        api_url = "http://localhost:8000/analyze-sentiment"
        
        try:
            # Handle batch analysis
            if isinstance(text, list):
                results = []
                for i, single_text in enumerate(text):
                    try:
                        payload = {"text": single_text}
                        response = requests.post(
                            api_url, 
                            json=payload, 
                            headers={"Content-Type": "application/json"},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            result["original_text"] = single_text
                            result["index"] = i
                            results.append(result)
                        else:
                            error_result = {
                                "index": i,
                                "original_text": single_text,
                                "error": f"API returned status code {response.status_code}",
                                "response": response.text
                            }
                            results.append(error_result)
                            
                    except requests.exceptions.RequestException as e:
                        error_result = {
                            "index": i,
                            "original_text": single_text,
                            "error": f"Request failed: {str(e)}"
                        }
                        results.append(error_result)
                
                return json.dumps({
                    "batch_results": results,
                    "total_analyzed": len(text),
                    "successful_analyses": len([r for r in results if "error" not in r])
                }, indent=2)
            
            # Handle single text analysis
            else:
                payload = {"text": text}
                response = requests.post(
                    api_url, 
                    json=payload, 
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    result["original_text"] = text
                    
                    return json.dumps({
                        "sentiment": result.get("sentiment", "unknown"),
                        "confidence": result.get("confidence", 0.0),
                        "scores": result.get("scores", {}),
                        "original_text": text,
                        "analysis_successful": True
                    }, indent=2)
                else:
                    return json.dumps({
                        "error": f"API returned status code {response.status_code}",
                        "response": response.text,
                        "original_text": text,
                        "analysis_successful": False
                    }, indent=2)
                    
        except requests.exceptions.ConnectionError:
            return json.dumps({
                "error": "Could not connect to FinBERT API at localhost:8000. Please ensure the API server is running.",
                "original_text": text if isinstance(text, str) else f"{len(text)} texts",
                "analysis_successful": False
            }, indent=2)
            
        except requests.exceptions.Timeout:
            return json.dumps({
                "error": "Request to FinBERT API timed out after 30 seconds.",
                "original_text": text if isinstance(text, str) else f"{len(text)} texts",
                "analysis_successful": False
            }, indent=2)
            
        except requests.exceptions.RequestException as e:
            return json.dumps({
                "error": f"Request failed: {str(e)}",
                "original_text": text if isinstance(text, str) else f"{len(text)} texts",
                "analysis_successful": False
            }, indent=2)
            
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": f"Failed to parse API response as JSON: {str(e)}",
                "original_text": text if isinstance(text, str) else f"{len(text)} texts",
                "analysis_successful": False
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Unexpected error occurred: {str(e)}",
                "original_text": text if isinstance(text, str) else f"{len(text)} texts",
                "analysis_successful": False
            }, indent=2)