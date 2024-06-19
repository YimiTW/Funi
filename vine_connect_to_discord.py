import discord
import vine
import os

vine.mode = 'discord'

# client是跟discord連接，intents是要求機器人的權限
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

from dotenv import load_dotenv

load_dotenv()

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

    vine_response = vine.main_request(input_text, display_name, vine.chat_data_all_path)
    if vine_response == "<skip>":
        print(f"\n藤: skip")
    elif len(vine_response) > 2000:
        msg1 = vine_response[:2000]
        await message.channel.send(msg1)
        msg2 = vine_response.replace(msg1, '')
        await message.channel.send(msg2)
    else:
        await message.channel.send(vine_response)

client.run(os.getenv('token'))
