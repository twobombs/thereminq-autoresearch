import discord
import os
import requests
import asyncio

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

def query_local_llm(user_message):
    # This points to your local llama-server running on port 8033
    url = "http://127.0.0.1:8033/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "messages": [
            # You can change this system prompt to give your bot a specific personality!
            {"role": "system", "content": "You are a helpful, conversational AI assistant hanging out in a Discord server. Keep your answers concise and friendly."},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    # Extract the text from the JSON response
    return response.json()['choices'][0]['message']['content']

@client.event
async def on_message(message):
    # Ignore the message if the bot isn't mentioned, or if it's from the bot itself
    if client.user not in message.mentions:
        return
    if message.author == client.user:
        return

    # Show the "ThereminQ-bot is typing..." indicator in Discord
    async with message.channel.typing():
        try:
            # Strip the @Bot mention out of the text so the AI doesn't get confused by the ID tag
            clean_content = message.content.replace(f'<@{client.user.id}>', '').strip()
            
            # Send the message to your local GPU (using a thread so it doesn't freeze the Discord connection)
            llm_reply = await asyncio.to_thread(query_local_llm, clean_content)
            
            # Send the generated response back to Discord!
            await message.reply(llm_reply)
            
        except requests.exceptions.ConnectionError:
            await message.reply("My brain is offline! Is `llama-server` running on port 8033?")
        except Exception as e:
            await message.reply(f"Oops, I hit a snag: {e}")

client.run(os.environ.get('DISCORD_BOT_TOKEN'))
