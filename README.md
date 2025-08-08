# Jarvis AI Assistant

A Python-based personal voice assistant that can listen to your commands, answer questions, fetch weather reports, read news, open websites, and much more using multiple libraries.

## Features
- Voice recognition and speech output
- Weather updates
- News headlines
- Web search
- Open applications
- Jokes, Wikipedia, and more
- Modular structure for adding more commands

## Project Structure
jarvis-ai/
    ├── jarvis.py                # Main program file
    ├── config.py                # API keys and configuration
    ├── requirements.txt         # Required Python libraries
    └── modules/                 # Extra command modules

## Prerequisites
- Python 3.8 or above
- Microphone and speaker
- Internet connection

## Installation and Setup

Step 1 – Clone the Repository:
git clone https://github.com/yourusername/jarvis-ai.git
cd jarvis-ai

Step 2 – Install Dependencies:
pip install -r requirements.txt

Step 3 – Add API Keys:
Create a file named config.py and add the following:
WEATHER_API_KEY = "your_weather_api_key"
NEWS_API_KEY = "your_news_api_key"
GPT_API_KEY = "your_openai_api_key"

You can get free API keys from:
- Weather API: https://openweathermap.org/api
- News API: https://newsapi.org
- GPT API: https://platform.openai.com

Step 4 – Run the Assistant:
python jarvis.py

## Example Commands
- "Jarvis, what’s the weather today?"
- "Tell me the news."
- "Search for Python tutorials."
- "Open YouTube."
- "Tell me a joke."
- "Exit."

## requirements.txt Example
speechrecognition
pyttsx3
pyaudio
requests
wikipedia
openai
python-dotenv
newsapi-python

## License
This project is licensed under the MIT License.

## Author
**Name:** Souvik Kumar  
**GitHub:** [https://github.com/souvik5169](https://github.com/souvik5169)  
**Email:** cdtsouvikkumar@gmail.com
