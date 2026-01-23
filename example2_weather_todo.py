import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os 
from dotenv import load_dotenv   
from autogen_agentchat.ui import Console
import requests
load_dotenv()
 
#langsmith integration
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")

async def get_weather(city:str)->str:
    """Get the weather for a given city."""
    try:
        API_KEY=os.getenv("OPEN_WEATHER_API_KEY")
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response=requests.get(url,timeout=60)
        data=response.json()
        temp=data['main']['temp']
        description=data['weather'][0]['description']
        return f"{city}: {temp} !C, {description}"

    except Exception as e:
        return f"Error: {str(e)}"
 
async def main() -> None:
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    agent=AssistantAgent(
        name="King",   
        model_client=model_client,   
        tools=[get_weather],       
        system_message="You are a helpful assistant.",      
        model_client_stream=True,        
    )   

    while True:
        city=input("Enter city name(or 'exit): ").strip()
        if city.lower()=="exit":
            break

        response=agent.run_stream(task=f"""
        What is the weather in {city}?.You should not Explain anyother thing. if city is not valid just give 
        Invalid city and no further conversation""")
        await Console(response)
        print("-"*40)
    await model_client.close()

asyncio.run(main())
 