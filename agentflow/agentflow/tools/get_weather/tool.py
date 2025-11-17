import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

from agentflow.tools.base import BaseTool

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Get_Weather_Tool"

LIMITATIONS = """
1. This tool requires a valid OpenWeatherMap API key.
2. Weather data is updated every 10-15 minutes, so very recent changes may not be reflected.
3. The tool provides current weather data only (not historical or long-term forecasts beyond 5 days).
4. Location accuracy depends on the OpenWeatherMap geocoding service.
"""

BEST_PRACTICES = """
1. Use specific location names (city, state/country) for better accuracy, e.g., "Paris, France" or "London, UK".
2. For cities with common names, include the country code or state, e.g., "Springfield, IL, US".
3. The tool returns current weather conditions including temperature, humidity, wind speed, and weather description.
4. Temperature is returned in Celsius by default, but you can specify units (metric, imperial, or standard Kelvin).
5. This tool is ideal for getting real-time weather information for location-based queries.
"""

class Get_Weather_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A weather information tool that provides real-time weather data for any location worldwide using OpenWeatherMap API.",
            tool_version="1.0.0",
            input_types={
                "location": "str - The location to get weather information for (city name, city with country, coordinates).",
                "units": "str - Temperature units: 'metric' (Celsius), 'imperial' (Fahrenheit), or 'standard' (Kelvin). Default is 'metric'.",
            },
            output_type="dict - A dictionary containing weather information including temperature, conditions, humidity, wind speed, and more.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(location="London, UK")',
                    "description": "Get current weather for London, UK with default metric units (Celsius)."
                },
                {
                    "command": 'execution = tool.execute(location="New York, US", units="imperial")',
                    "description": "Get current weather for New York with imperial units (Fahrenheit)."
                },
                {
                    "command": 'execution = tool.execute(location="Tokyo, Japan")',
                    "description": "Get current weather for Tokyo, Japan."
                }
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            }
        )
        self.max_retries = 3
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
        # Get API key from environment
        self.api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
        if not self.api_key:
            raise Exception(
                "OpenWeatherMap API key not found. Please set the OPENWEATHERMAP_API_KEY environment variable. "
                "You can get a free API key at https://openweathermap.org/api"
            )

    def _get_weather_data(self, location: str, units: str = "metric"):
        """
        Fetch weather data from OpenWeatherMap API.
        
        Args:
            location (str): The location to get weather for.
            units (str): Temperature units - 'metric', 'imperial', or 'standard'.
            
        Returns:
            dict: Weather data or error information.
        """
        # Prepare API request parameters
        params = {
            "q": location,
            "appid": self.api_key,
            "units": units
        }
        
        for attempt in range(self.max_retries):
            try:
                # Make the API call
                response = requests.get(self.base_url, params=params, timeout=10)
                
                # Check if request was successful
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return {"error": f"Location '{location}' not found. Please check the location name and try again."}
                elif response.status_code == 401:
                    return {"error": "Invalid API key. Please check your OPENWEATHERMAP_API_KEY."}
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
        
        return {"error": "Failed to fetch weather data after multiple retries."}

    def _format_weather_response(self, data: dict, units: str):
        """
        Format the weather data into a readable response.
        
        Args:
            data (dict): Raw weather data from API.
            units (str): Temperature units used.
            
        Returns:
            dict: Formatted weather information.
        """
        if "error" in data:
            return data
        
        # Determine temperature unit symbol
        temp_unit = "°C" if units == "metric" else ("°F" if units == "imperial" else "K")
        speed_unit = "m/s" if units == "metric" else ("mph" if units == "imperial" else "m/s")
        
        # Extract relevant information
        weather_info = {
            "location": f"{data.get('name', 'Unknown')}, {data.get('sys', {}).get('country', 'Unknown')}",
            "coordinates": {
                "latitude": data.get('coord', {}).get('lat'),
                "longitude": data.get('coord', {}).get('lon')
            },
            "weather": {
                "condition": data.get('weather', [{}])[0].get('main', 'Unknown'),
                "description": data.get('weather', [{}])[0].get('description', 'Unknown'),
                "icon": data.get('weather', [{}])[0].get('icon', '')
            },
            "temperature": {
                "current": f"{data.get('main', {}).get('temp', 'N/A')}{temp_unit}",
                "feels_like": f"{data.get('main', {}).get('feels_like', 'N/A')}{temp_unit}",
                "min": f"{data.get('main', {}).get('temp_min', 'N/A')}{temp_unit}",
                "max": f"{data.get('main', {}).get('temp_max', 'N/A')}{temp_unit}"
            },
            "atmospheric_conditions": {
                "pressure": f"{data.get('main', {}).get('pressure', 'N/A')} hPa",
                "humidity": f"{data.get('main', {}).get('humidity', 'N/A')}%",
                "visibility": f"{data.get('visibility', 'N/A')} meters"
            },
            "wind": {
                "speed": f"{data.get('wind', {}).get('speed', 'N/A')} {speed_unit}",
                "direction": f"{data.get('wind', {}).get('deg', 'N/A')}°"
            },
            "clouds": f"{data.get('clouds', {}).get('all', 'N/A')}%",
            "timestamp": data.get('dt', 'N/A'),
            "timezone": f"UTC{data.get('timezone', 0) // 3600:+d}"
        }
        
        # Add rain/snow information if available
        if 'rain' in data:
            weather_info['rain'] = data['rain']
        if 'snow' in data:
            weather_info['snow'] = data['snow']
        
        return weather_info

    def execute(self, location: str, units: str = "metric"):
        """
        Execute the weather tool to get current weather information.

        Parameters:
            location (str): The location to get weather information for.
            units (str): Temperature units - 'metric' (Celsius), 'imperial' (Fahrenheit), or 'standard' (Kelvin).
                        Default is 'metric'.

        Returns:
            dict: A dictionary containing weather information or error message.
        """
        # Validate units parameter
        valid_units = ["metric", "imperial", "standard"]
        if units not in valid_units:
            return {"error": f"Invalid units '{units}'. Must be one of: {', '.join(valid_units)}"}
        
        # Fetch weather data
        weather_data = self._get_weather_data(location, units)
        
        # Format and return the response
        formatted_response = self._format_weather_response(weather_data, units)
        
        return formatted_response

    def get_metadata(self):
        """
        Returns the metadata for the Get_Weather_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    """
    Test:
    cd agentflow/tools/get_weather
    python tool.py
    """
    def print_json(result):
        import json
        print(json.dumps(result, indent=4, ensure_ascii=False))

    # Check if API key is set
    if not os.environ.get("OPENWEATHERMAP_API_KEY"):
        print("ERROR: OPENWEATHERMAP_API_KEY not set!")
        print("Get a free API key at: https://openweathermap.org/api")
        print("Then set it with: export OPENWEATHERMAP_API_KEY='your_key_here'")
        exit(1)

    weather_tool = Get_Weather_Tool()

    # Get tool metadata
    metadata = weather_tool.get_metadata()
    print("Tool Metadata:")
    print_json(metadata)
    print("\n" + "="*80 + "\n")

    examples = [
        {'location': 'London, UK'},
        # {'location': 'New York, US', 'units': 'imperial'},
        # {'location': 'Tokyo, Japan'},
        # {'location': 'Paris, France', 'units': 'metric'},
    ]
    
    for example in examples:
        print(f"Executing weather query: {example['location']}")
        if 'units' in example:
            print(f"Units: {example['units']}")
        try:
            result = weather_tool.execute(**example)
            print("Weather Result:")
            print_json(result)
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 80 + "\n")

    print("Done!")