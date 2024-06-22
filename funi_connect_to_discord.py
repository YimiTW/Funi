import discord
import os
import time

import funi
# shitch to discord mode
funi.mode = 'discord'

from dotenv import load_dotenv
# load .env
load_dotenv()

# client是跟discord連接，intents是要求機器人的權限
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)



# 調用event函式庫
@client.event
# 當機器人完成啟動
async def on_ready():
    print(f"目前登入身份 --> {client.user}")

@client.event
# 當頻道有新訊息
async def on_message(message):
    # 排除機器人本身的訊息，避免無限循環
    if message.author == client.user:
        return
    
    username = message.author.name
    user_id = message.author.id
    display_name = message.author.display_name
    input_text = message.content

    # print user message
    print(f"\n{display_name}: {input_text}")

    funi_response = funi.main_request(input_text, display_name)

    if funi_response != "*skip*":
        await message.channel.send(funi_response)

client.run(os.getenv('token'))
