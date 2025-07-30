import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.agents import initialize_agent, AgentType, Tool

from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# Define system message to act as a research assistant
system_message = SystemMessagePromptTemplate.from_template(
    "You are a senior research assistant. Provide in-depth, well-structured, and citation-rich answers. "
    "Use Tavily search for supporting evidence. Think step-by-step. Avoid vague responses."
)

# Prompt Template with memory
prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Setup Tavily tool
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY)
tools = [Tool.from_function(func=tavily_tool.run, name="Tavily Search", description="Search for recent or scholarly info")]

# App
app = FastAPI()

# Request body
class QueryRequest(BaseModel):
    session_id: str
    query: str

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    # Set Redis memory per session
    chat_history = RedisChatMessageHistory(
        session_id=request.session_id,
        url=REDIS_URL
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_history,
        return_messages=True
    )

    # Create the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={"prompt": prompt}
    )

    # Ask original query
    response = agent.invoke(request.query)
    content = response.content.strip() if hasattr(response, "content") else str(response)

    # Reflection Loop: retry if answer too short or generic
    if len(content.split()) < 50 or "I'm not sure" in content:
        followup_prompt = f"Please go deeper and add more specific sources for: {request.query}"
        response = agent.invoke(followup_prompt)
        content = response.content.strip() if hasattr(response, "content") else str(response)

    return JSONResponse(content={"response": content})