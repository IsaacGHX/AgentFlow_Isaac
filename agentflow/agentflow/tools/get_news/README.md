# Get Weather Tool

A weather information tool that provides real-time weather data for any location worldwide using the OpenWeatherMap API.

## Features

- **Real-time weather data**: Get current weather conditions for any location
- **Multiple temperature units**: Support for Celsius, Fahrenheit, and Kelvin
- **Comprehensive information**: Temperature, humidity, wind speed, atmospheric pressure, visibility, and more
- **Global coverage**: Works for any location worldwide
- **Error handling**: Robust retry mechanism and clear error messages

## Setup

### 1. Install Dependencies

```bash
pip install requests python-dotenv
```

Or install all project requirements:

```bash
pip install -r requirements.txt
```

### 2. Get OpenWeatherMap API Key

1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Get your API key from the dashboard

### 3. Set Environment Variable

```bash
export OPENWEATHERMAP_API_KEY='your_api_key_here'
```

Or add it to your `.env` file:

```
OPENWEATHERMAP_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
from agentflow.tools.get_weather import Get_Weather_Tool

# Initialize the tool
weather_tool = Get_Weather_Tool()

# Get weather for a location (default: Celsius)
result = weather_tool.execute(location="London, UK")
print(result)
```

### With Different Units

```python
# Get weather in Fahrenheit
result = weather_tool.execute(location="New York, US", units="imperial")

# Get weather in Kelvin
result = weather_tool.execute(location="Tokyo, Japan", units="standard")
```

### Example Output

```json
{
    "location": "London, GB",
    "coordinates": {
        "latitude": 51.5074,
        "longitude": -0.1278
    },
    "weather": {
        "condition": "Clouds",
        "description": "overcast clouds",
        "icon": "04d"
    },
    "temperature": {
        "current": "15.2°C",
        "feels_like": "14.8°C",
        "min": "13.5°C",
        "max": "16.7°C"
    },
    "atmospheric_conditions": {
        "pressure": "1015 hPa",
        "humidity": "72%",
        "visibility": "10000 meters"
    },
    "wind": {
        "speed": "3.5 m/s",
        "direction": "220°"
    },
    "clouds": "90%",
    "timestamp": 1700237400,
    "timezone": "UTC+0"
}
```

## Testing

Run the test script:

```bash
cd agentflow/tools/get_weather
python tool.py
```

## Best Practices

1. **Use specific location names** for better accuracy: `"Paris, France"` or `"London, UK"`
2. **Include country codes** for common city names: `"Springfield, IL, US"`
3. **Choose appropriate units**:
   - `metric` for Celsius (default)
   - `imperial` for Fahrenheit
   - `standard` for Kelvin

## Limitations

1. Requires a valid OpenWeatherMap API key
2. Weather data is updated every 10-15 minutes
3. Provides current weather only (not historical data)
4. Location accuracy depends on OpenWeatherMap's geocoding service

## Error Handling

The tool provides clear error messages for common issues:

- **Location not found**: Check the location name spelling
- **Invalid API key**: Verify your `OPENWEATHERMAP_API_KEY`
- **Network timeout**: The tool automatically retries (up to 3 times)
- **Invalid units**: Must be `metric`, `imperial`, or `standard`

## API Key Limits

Free tier OpenWeatherMap API includes:
- 1,000 API calls per day
- 60 calls per minute
- Current weather data access

For higher limits, consider upgrading your OpenWeatherMap plan.

