# import os
# from dotenv import load_dotenv
# from livekit import agents
# from livekit.agents import AgentSession, Agent, JobContext, WorkerOptions, cli
# from livekit.plugins import openai, silero 

# load_dotenv()
 
# os.environ["LIVEKIT_URL"] = "wss://sreddy-7x44jwpl.livekit.cloud"
# os.environ["LIVEKIT_API_KEY"] = "APIMCRUQLMMLWd6"
# os.environ["LIVEKIT_API_SECRET"] = "KbJXUOSCGIS8x5IVdBUufQ4vSUfvCDEirvxi1cjTowo"

# async def entrypoint(ctx: JobContext):
#     await ctx.connect()
#     print(f"Voice Agent connected to room: {ctx.room.name}")
     
#     session = AgentSession(
#         vad=silero.VAD.load(),  
#         stt=openai.STT(),
#         llm=openai.LLM(),
#         tts=openai.TTS(),
#     )
     
#     await session.start(
#         room=ctx.room,
#         agent=Agent(
#             instructions="You are a helpful assistant. Keep your responses under two sentences."
#         )
#     )
     
#     await session.generate_reply(
#         instructions="Greet the user by saying: Hey bo..! The LiveKit and Exotel bridge is working fine."
#     )

# if __name__ == "__main__":
#     cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


import os
import aiohttp  
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe, AgentSession, Agent, llm
from livekit.plugins import azure, openai, silero 
load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

def sanitize_for_tts(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    text = text.replace("&", "and")
    text = text.replace("<", "")
    text = text.replace(">", "")
    return text

class LangGraphAgent(Agent):
    def __init__(self, room_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.room_name = "1234"
        self.first_greeting = "Hello, I am connecting to the server." # Fallback

    # 1. INITIALIZE THE MEMORY ON THE BACKEND
    async def setup_backend(self):
        url = "https://oliver-patulous-valeria.ngrok-free.dev/agent/init"
        payload = {
            "session_id": str(self.room_name),
            "lead_name": "Reddy",
            "phone": "+910000000000",
            "lead_id": "LEAD001",
            "email": "test@example.com"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Save the real greeting from LangGraph!
                        self.first_greeting = data.get("agent_text", "Hello, how can I help you today?")
                        print(" Backend Initialized Successfully!")
                    else:
                        print(f" Backend Init Failed: HTTP {resp.status}")
        except Exception as e:
            print(f" Failed to reach init endpoint: {e}")

    # 2. HANDLE THE CONVERSATION TURNS
    async def llm_node(self, chat_ctx: llm.ChatContext, tools: list, model_settings=None):
        
        message_list = chat_ctx.messages() if callable(chat_ctx.messages) else chat_ctx.messages
        user_msg = message_list[-1].content
        
        if isinstance(user_msg, list):
            user_msg = user_msg[0]
             
        # Trigger the initial greeting fetched from the /init endpoint!
        if "Greet the user" in str(user_msg):
            yield sanitize_for_tts(self.first_greeting)
            return

        print(f"\n[Customer]: {user_msg}")
        
        url = "https://oliver-patulous-valeria.ngrok-free.dev/agent/turn" 
        payload = {
            "session_id": str(self.room_name),
            "user_text": str(user_msg)
        }
        
        try:
            # Hit the Turn endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        bot_reply_text = data.get("agent_text", "Sorry, I received an empty response.")
                        
                        # Optional: If the bot decides to hang up
                        if data.get("call_ended") is True:
                            print(" Bot requested to end the call.")
                            
                    else:
                        bot_reply_text = f"Sorry, the server returned an error code {resp.status}."
                        
        except Exception as e:
            print(f"API request failed: {e}")
            bot_reply_text = "Sorry, my logic engine lost connection."
 
        # Sanitize and Yield to Azure
        safe_reply_text = sanitize_for_tts(bot_reply_text)
        print(f"[Agent]: {safe_reply_text}")
        yield safe_reply_text

async def entrypoint(ctx: JobContext): 
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    print(f"Voice Agent connected to room: {ctx.room.name}")
      
    session = AgentSession(
        vad=silero.VAD.load(),  
        stt=azure.STT(), 
        tts=azure.TTS(voice="en-IN-NeerjaNeural"), 
        llm=openai.LLM() 
    )
  
    agent = LangGraphAgent(room_name=ctx.room.name, instructions="")
    
    # 🚨 CRITICAL: Call the /init endpoint BEFORE starting the call!
    await agent.setup_backend()
    
    await session.start(room=ctx.room, agent=agent)
       
    # This triggers LiveKit to pull the greeting we saved during setup_backend
    await session.generate_reply(
        instructions="Greet the user by saying the initial greeting."
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))