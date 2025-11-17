import os
import json
from dotenv import load_dotenv
load_dotenv()

from perplexity import Perplexity

from agentflow.tools.base import BaseTool

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Perplexity_Search_Tool"

LIMITATIONS = """
1. This tool is only suitable for general information search.
2. This tool contains less domain specific information.
3. This tools is not suitable for searching and analyzing videos at YouTube or other video platforms.
"""

BEST_PRACTICES = """
1. Choose this tool when you want to search general information about a topic.
2. Choose this tool for question type of query, such as "What is the capital of France?" or "What is the capital of France?"
3. The tool will return a summarized information.
4. This tool is more suitable for definition, world knowledge, and general information search.
"""

class Perplexity_Search_Tool(BaseTool):
    def __init__(self, model_string="sonar"):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A web search tool powered by Perplexity AI that provides real-time information from the internet.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query to find information on the web.",
            },
            output_type="str - The search results of the query.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the capital of France?")',
                    "description": "Search for general information about the capital of France."
                },
                {
                    "command": 'execution = tool.execute(query="Who won the euro 2024?")',
                    "description": "Search for information about Euro 2024 winner."
                },
                {
                    "command": 'execution = tool.execute(query="Physics and Society article arXiv August 11, 2016")',
                    "description": "Search for specific academic articles."
                }
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            }
        )
        self.max_retries = 5
        self.search_model = model_string

        try:
            # Initialize the client (uses PERPLEXITY_API_KEY environment variable)
            self.client = Perplexity(api_key=os.environ.get("PERPLEXITY_API_KEY"))
        except Exception as e:
            raise Exception(f"Perplexity API key not found. Please set the PERPLEXITY_API_KEY environment variable. Error: {str(e)}")


    def _execute_search(self, query: str):
        """
        Execute a search using Perplexity API.
        
        Args:
            query (str): The search query to find information on the web.
            
        Returns:
            str: The search results from Perplexity.
        """
        response_text = None
        
        for attempt in range(self.max_retries):
            try:
                # Make the API call
                completion = self.client.chat.completions.create(
                    model=self.search_model,
                    messages=[
                        {"role": "user", "content": query}
                    ]
                )
                
                # Extract the response
                response_text = completion.choices[0].message.content
                
                # If we get here, the API call was successful, so break out of the retry loop
                break
            except Exception as e:
                print(f"Perplexity Search attempt {attempt + 1} failed: {str(e)}. Retrying...")
                if attempt == self.max_retries - 1:  # Last attempt
                    print(f"Perplexity Search failed after {self.max_retries} attempts. Last error: {str(e)}")
                    return f"Perplexity Search tried {self.max_retries} times but failed. Last error: {str(e)}"
                # Continue to next attempt

        # Check if we have a valid response before proceeding
        if response_text is None:
            return "Perplexity Search failed to get a valid response"

        return response_text

    def execute(self, query: str):
        """
        Execute the Perplexity search tool.

        Parameters:
            query (str): The search query to find information on the web.

        Returns:
            str: The search results of the query.
        """
        # Perform the search
        response = self._execute_search(query)
        
        return response

    def get_metadata(self):
        """
        Returns the metadata for the Perplexity_Search tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    """
    Test:
    cd agentflow/tools/perplexity_search
    python tool.py
    """
    def print_json(result):
        import json
        print(json.dumps(result, indent=4))

    perplexity_search = Perplexity_Search_Tool()

    # Get tool metadata
    metadata = perplexity_search.get_metadata()
    print("Tool Metadata:")
    print_json(metadata)

    examples = [
        # {'query': 'What is the capital of France?'},
        {'query': 'Who won the euro 2024?'},
        # {'query': 'Physics and Society article arXiv August 11, 2016'},
    ]
    
    for example in examples:
        print(f"\nExecuting search: {example['query']}")
        try:
            result = perplexity_search.execute(**example)
            print("Search Result:")
            print(result)
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)

    print("Done!")