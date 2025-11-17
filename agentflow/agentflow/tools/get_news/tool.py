import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from agentflow.tools.base import BaseTool

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Get_News_Tool"

LIMITATIONS = """
1. This tool requires a valid NewsAPI API key (get one free at https://newsapi.org).
2. Free tier has limitations: 100 requests per day, articles up to 1 month old.
3. Results are limited to the specified number of articles (max 100 per request).
4. Some article content may be truncated - full content requires visiting the source URL.
5. Real-time news updates depend on when sources publish and NewsAPI indexes them.
"""

BEST_PRACTICES = """
1. Use specific keywords or phrases for more relevant results, e.g., "artificial intelligence", "climate change".
2. Specify a category (business, technology, sports, etc.) to narrow down results.
3. Use country codes (us, gb, ca, etc.) to get news from specific countries.
4. Combine query with category and country for highly targeted results.
5. For trending topics, use 'top-headlines' mode; for specific searches, use 'everything' mode.
6. Specify language (en, es, fr, etc.) to get news in your preferred language.
"""

class Get_News_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A news retrieval tool that provides real-time news articles and top headlines from various sources worldwide using NewsAPI.",
            tool_version="1.0.0",
            input_types={
                "query": "str - Search keywords or phrases to find relevant news articles. Optional for top-headlines mode.",
                # "mode": "str - 'top-headlines' for breaking news or 'everything' for comprehensive search. Default is 'top-headlines'.",
                # "category": "str - News category: business, entertainment, general, health, science, sports, technology. Only for top-headlines mode.",
                # "country": "str - 2-letter ISO 3166-1 country code (e.g., 'us', 'gb', 'ca'). Only for top-headlines mode.",
                # "language": "str - 2-letter ISO 639-1 language code (e.g., 'en', 'es', 'fr'). Default is 'en'.",
                # "page_size": "int - Number of articles to return (max 100). Default is 10.",
                # "sort_by": "str - Sort order: 'relevancy', 'popularity', or 'publishedAt'. Only for everything mode. Default is 'publishedAt'.",
            },
            output_type="dict - A dictionary containing news articles with title, description, content, source, author, URL, and publication date.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="artificial intelligence", mode="everything")',
                    "description": "Search for all news articles about artificial intelligence."
                },
                # {
                #     "command": 'execution = tool.execute(category="technology", country="us")',
                #     "description": "Get top technology headlines from the United States."
                # },
                # {
                #     "command": 'execution = tool.execute(query="climate change", mode="everything", language="en", page_size=5)',
                #     "description": "Search for 5 recent English news articles about climate change."
                # },
                # {
                #     "command": 'execution = tool.execute(category="business")',
                #     "description": "Get top business headlines."
                # }
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            }
        )
        self.max_retries = 3
        self.top_headlines_url = "https://newsapi.org/v2/top-headlines"
        self.everything_url = "https://newsapi.org/v2/everything"
        
        # Get API key from environment
        self.api_key = os.environ.get("NEWSAPI_KEY")
        if not self.api_key:
            raise Exception(
                "NewsAPI key not found. Please set the NEWSAPI_KEY environment variable. "
                "You can get a free API key at https://newsapi.org/register"
            )

    def _get_news_data(self, mode: str, query: str = None, category: str = None, 
                       country: str = None, language: str = "en", page_size: int = 5, 
                       sort_by: str = "publishedAt"):
        """
        Fetch news data from NewsAPI.
        
        Args:
            mode (str): 'top-headlines' or 'everything'.
            query (str): Search keywords.
            category (str): News category (for top-headlines).
            country (str): Country code (for top-headlines).
            language (str): Language code.
            page_size (int): Number of articles to return.
            sort_by (str): Sort order (for everything mode).
            
        Returns:
            dict: News data or error information.
        """
        # Select appropriate endpoint
        url = self.top_headlines_url if mode == "top-headlines" else self.everything_url
        
        # Prepare API request parameters
        params = {
            "apiKey": self.api_key,
            "pageSize": min(page_size, 100),  # NewsAPI max is 100
        }
        
        # Add mode-specific parameters
        if mode == "top-headlines":
            if query:
                params["q"] = query
            if category:
                params["category"] = category
            if country:
                params["country"] = country
            elif not query and not category:
                # Default to US if no query, category, or country specified
                params["country"] = "us"
        else:  # everything mode
            if not query:
                return {"error": "Query parameter is required for 'everything' mode."}
            params["q"] = query
            params["sortBy"] = sort_by
            if language:
                params["language"] = language
        
        for attempt in range(self.max_retries):
            try:
                # Make the API call
                response = requests.get(url, params=params, timeout=15)
                
                # Check if request was successful
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    return {"error": "Invalid API key. Please check your NEWSAPI_KEY."}
                elif response.status_code == 426:
                    return {"error": "API key tier limitation. This feature may require a paid plan."}
                elif response.status_code == 429:
                    return {"error": "Rate limit exceeded. Please try again later."}
                elif response.status_code == 400:
                    error_msg = response.json().get('message', 'Bad request')
                    return {"error": f"Bad request: {error_msg}"}
                else:
                    return {"error": f"API request failed with status code {response.status_code}: {response.text}"}
                    
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    return {"error": f"Request timed out after {self.max_retries} attempts."}
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    return {"error": f"Request failed after {self.max_retries} attempts. Error: {str(e)}"}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}"}
        
        return {"error": "Failed to fetch news data after multiple retries."}

    def _format_news_response(self, data: dict):
        """
        Format the news data into a readable response.
        
        Args:
            data (dict): Raw news data from API.
            
        Returns:
            dict: Formatted news information.
        """
        if "error" in data:
            return data
        
        # Check API response status
        if data.get('status') != 'ok':
            error_message = data.get('message', 'Unknown error')
            return {"error": f"API returned error status: {error_message}"}
        
        # Extract articles
        articles = data.get('articles', [])
        total_results = data.get('totalResults', 0)
        
        if not articles:
            return {
                "status": "success",
                "total_results": 0,
                "articles_returned": 0,
                "articles": [],
                "message": "No articles found matching your criteria."
            }
        
        # Format each article
        formatted_articles = []
        for article in articles:
            formatted_article = {
                "title": article.get('title', 'No title'),
                "description": article.get('description', 'No description available'),
                "content": article.get('content', 'Content not available'),
                "author": article.get('author', 'Unknown author'),
                "source": {
                    "name": article.get('source', {}).get('name', 'Unknown source'),
                    "id": article.get('source', {}).get('id')
                },
                "url": article.get('url', ''),
                "image_url": article.get('urlToImage'),
                "published_at": article.get('publishedAt', ''),
                "published_date": self._format_date(article.get('publishedAt', ''))
            }
            formatted_articles.append(formatted_article)
        
        return {
            "status": "success",
            "total_results": total_results,
            "articles_returned": len(formatted_articles),
            "articles": formatted_articles
        }
    
    def _format_date(self, date_string: str):
        """
        Format ISO date string to readable format.
        
        Args:
            date_string (str): ISO format date string.
            
        Returns:
            str: Formatted date string.
        """
        if not date_string:
            return "Unknown date"
        
        try:
            dt = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
            return dt.strftime("%B %d, %Y at %I:%M %p UTC")
        except:
            return date_string

    def execute(self, query: str = None, mode: str = "top-headlines", category: str = None,
                country: str = None, language: str = "en", page_size: int = 5, 
                sort_by: str = "publishedAt"):
        """
        Execute the news tool to get news articles.

        Parameters:
            query (str): Search keywords or phrases. Optional for top-headlines, required for everything mode.
            mode (str): 'top-headlines' for breaking news or 'everything' for comprehensive search. Default is 'top-headlines'.
            category (str): News category (business, entertainment, general, health, science, sports, technology). 
                          Only for top-headlines mode.
            country (str): 2-letter ISO 3166-1 country code (e.g., 'us', 'gb', 'ca'). Only for top-headlines mode.
            language (str): 2-letter ISO 639-1 language code (e.g., 'en', 'es', 'fr'). Default is 'en'.
            page_size (int): Number of articles to return (max 100). Default is 10.
            sort_by (str): Sort order for everything mode: 'relevancy', 'popularity', or 'publishedAt'. 
                          Default is 'publishedAt'.

        Returns:
            dict: A dictionary containing news articles or error message.
        """
        # Validate mode parameter
        valid_modes = ["top-headlines", "everything"]
        if mode not in valid_modes:
            return {"error": f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}"}
        
        # Validate category parameter
        valid_categories = ["business", "entertainment", "general", "health", "science", "sports", "technology"]
        if category and category not in valid_categories:
            return {"error": f"Invalid category '{category}'. Must be one of: {', '.join(valid_categories)}"}
        
        # Validate sort_by parameter
        valid_sort_options = ["relevancy", "popularity", "publishedAt"]
        if sort_by not in valid_sort_options:
            return {"error": f"Invalid sort_by '{sort_by}'. Must be one of: {', '.join(valid_sort_options)}"}
        
        # Validate page_size
        if page_size < 1 or page_size > 100:
            return {"error": "page_size must be between 1 and 100."}
        
        # Fetch news data
        news_data = self._get_news_data(
            mode=mode,
            query=query,
            category=category,
            country=country,
            language=language,
            page_size=page_size,
            sort_by=sort_by
        )
        
        # Format and return the response
        formatted_response = self._format_news_response(news_data)
        
        return formatted_response

    def get_metadata(self):
        """
        Returns the metadata for the Get_News_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    """
    Test:
    cd agentflow/tools/get_news
    python tool.py
    """
    def print_json(result):
        import json
        print(json.dumps(result, indent=4, ensure_ascii=False))

    # Check if API key is set
    if not os.environ.get("NEWSAPI_KEY"):
        print("ERROR: NEWSAPI_KEY not set!")
        print("Get a free API key at: https://newsapi.org/register")
        print("Then set it with: export NEWSAPI_KEY='your_key_here'")
        exit(1)

    news_tool = Get_News_Tool()

    # Get tool metadata
    metadata = news_tool.get_metadata()
    print("Tool Metadata:")
    print_json(metadata)
    print("\n" + "="*80 + "\n")

    examples = [
        # {'category': 'technology', 'page_size': 3},
        # {'query': 'artificial intelligence', 'mode': 'everything', 'page_size': 3},
        # {'category': 'business', 'country': 'us', 'page_size': 5},
        # {'query': 'climate change', 'mode': 'everything', 'language': 'en', 'page_size': 3},
        {'query': 'artificial intelligence'},
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: Executing news query with parameters: {example}")
        try:
            result = news_tool.execute(**example)
            print("News Result:")
            print_json(result)
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 80 + "\n")

    print("Done!")