import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ✅ Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# ✅ Setup Tavily search
tavily_search = TavilySearchResults()

tools = [
    Tool(
        name="Tavily Search",
        func=tavily_search.run,
        description="Search the web using Tavily"
    )
]

# ✅ Setup Redis memory
chat_history = RedisChatMessageHistory(
    session_id="user_123",
    url=REDIS_URL
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=chat_history,
    return_messages=True
)

# ✅ Initialize the agent with memory
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# ✅ Reflection loop function
def reflection_loop_check_fn(result):
    if hasattr(result, "content"):
        return result.content.strip().upper()
    return str(result).strip().upper()

# ✅ Main interaction loop
if __name__ == "__main__":
    while True:
        query = input("💬 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = agent.invoke(query)
        print("\n🤖 Agent:", result.content.strip() if hasattr(result, "content") else result)