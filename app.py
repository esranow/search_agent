import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, PlainTextResponse
from dotenv import load_dotenv

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
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
    "You are a professional research assistant. Provide in-depth, well-structured, and citation-rich answers. "
    "Compare viewpoints, include examples, and reference sources where appropriate. "
    "Avoid vague language. When the user asks a question, treat it like a journalist preparing a report."
)

# Prompt Template with memory
prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Setup Tavily tool
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY)
tools = [
    Tool.from_function(
        func=tavily_tool.run,
        name="Tavily Search",
        description="Search for recent or scholarly info"
    )
]

# App
app = FastAPI(title="SageSync", description="Your intelligent assistant for deep research and personalized reporting.")

# Request body
class QueryRequest(BaseModel):
    session_id: str
    query: str

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    try:
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
        response = agent.invoke({"input": request.query})
        if isinstance(response, dict) and "output" in response:
            content = response["output"]
        else:
            content = str(response)

        # Reflect and re-ask if too short or vague
        if len(content.split()) < 100 or "i'm not sure" in content.lower() or "source" not in content.lower():
            followup_prompt = (
                f"Rewrite the answer to this query with greater depth, structured analysis, comparisons, and "
                f"add specific sources or citations if possible:\n\n{request.query}"
            )
            response = agent.invoke({"input": followup_prompt})
            if isinstance(response, dict) and "output" in response:
                content = response["output"]
            else:
                content = str(response)

        return JSONResponse(content={"response": content})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return PlainTextResponse(str(e), status_code=500)

@app.post("/report")
async def generate_report(request: QueryRequest):
    try:
        # Retrieve user chat history
        chat_history = RedisChatMessageHistory(
            session_id=request.session_id,
            url=REDIS_URL
        )
        messages = chat_history.messages

        if not messages:
            return JSONResponse(content={"error": "No history found for this session."}, status_code=404)

        # Aggregate user queries
        user_queries = [msg.content for msg in messages if msg.type == "human"]
        joined_queries = "\n".join(f"- {q}" for q in user_queries)

        # Build prompt to generate report
        report_prompt = (
            "You are an expert researcher. Based on the following list of user queries, generate a personalized, multi-section report. "
            "The report should include an introduction, key themes or areas of interest, in-depth explanations, comparisons, and cited sources. "
            "Use a formal yet friendly tone, and assume this report is for someone serious about learning or decision-making.\n\n"
            f"User's questions:\n{joined_queries}\n\n"
            "Generate the report accordingly."
        )

        # Run report generation
        report_response = llm.invoke(report_prompt)
        report_content = report_response.content.strip() if hasattr(report_response, "content") else str(report_response)

        return JSONResponse(content={"report": report_content})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return PlainTextResponse(str(e), status_code=500)
