import os
import toml
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from agent.tools import check_properties_tool

# --- 🔐 Load OpenAI API key from chainlit/secrets.toml ---
secrets_path = os.path.join(os.path.dirname(__file__), "chainlit", "secrets.toml")

try:
    secrets = toml.load(secrets_path)
    API_KEY = secrets.get("api_key")
    if not API_KEY or not API_KEY.startswith("sk-"):
        raise ValueError("Missing or invalid OpenAI API key.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load OpenAI API key: {e}")

# --- 🧠 Helper function to create the agent ---
def create_agent(api_key: str):
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        openai_api_key=api_key
    )
    return initialize_agent(
        tools=[check_properties_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

# --- ⚡ Chainlit startup event ---
@cl.on_chat_start
async def start():
    agent = create_agent(API_KEY)
    cl.user_session.set("agent", agent)

    await cl.Message(content="### 🏖️ **VacationGPT**", author=None).send()
    await cl.Message(
        content="Hi! I'm your travel assistant. Ask me anything about Airbnb properties in Sydney!",
        author="VacationGPT"
    ).send()

# --- 💬 Chainlit message handler ---
@cl.on_message
async def handle_message(msg: cl.Message):
    agent = cl.user_session.get("agent")

    if not agent:
        await cl.Message(
            content="❌ Agent not initialized. Please refresh and try again.",
            author="VacationGPT"
        ).send()
        return

    try:
        response = agent.run(msg.content)
    except Exception as e:
        response = f"❌ An error occurred while processing your request:\n```\n{str(e)}\n```"

    await cl.Message(content=response, author="VacationGPT").send()
