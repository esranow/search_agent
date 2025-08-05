import os
from fastapi import FastAPI, UploadFile, File
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

# File reading utils
import fitz  # PyMuPDF
import redis

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Redis client for RAG docs
redis_client = redis.from_url(REDIS_URL)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# Define system message
system_message = SystemMessagePromptTemplate.from_template(
    "You are a professional research assistant. Provide in-depth, well-structured, and citation-rich answers. "
    "Compare viewpoints, include examples, and reference sources where appropriate. "
    "Avoid vague language. When the user asks a question, treat it like a journalist preparing a report."
)

prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Tools
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY)
tools = [
    Tool.from_function(
        func=tavily_tool.run,
        name="Tavily Search",
        description="Search for recent or scholarly info"
    )
]

# FastAPI app
app = FastAPI(
    title="SageSync",
    description="""
SageSync is your intelligent research agent—designed not just to search, but to **understand, retain, and report**.

Unlike typical AI tools like Perplexity that only respond in the moment, SageSync is built for **long-term context**, **personalized memory**, and **multi-file RAG-based research**. Here’s what sets it apart:

- 🔍 **Context-Aware Search**: Go beyond Q&A—SageSync connects the dots across past queries, uploaded documents, and live questions.
- 🧠 **Personalized Memory**: It remembers what you’ve explored in the past to tailor better, deeper insights in future sessions.
- 📂 **Multimodal RAG Support**: Upload PDFs, text files, and more—SageSync grounds its answers in your actual content, not just the web.
- 📊 **Auto-Generated Research Reports**: At any point, generate a full research summary with references, highlights, and session insights.
- 🧩 **Session Intelligence**: Every session is more than chat—it’s a growing knowledge graph around your research journey.

Think of it as your **research brain extension**, not just another AI assistant.

"""
)


class QueryRequest(BaseModel):
    session_id: str
    query: str

@app.post("/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    try:
        content = await file.read()

        if file.filename.endswith(".pdf"):
            doc = fitz.open(stream=content, filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
        else:
            return PlainTextResponse("Only PDF files are supported right now.", status_code=400)

        redis_client.set(f"doc:{session_id}", text)
        return {"status": "success", "message": "File uploaded and processed."}

    except Exception as e:
        return PlainTextResponse(str(e), status_code=500)

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    try:
        # Redis chat memory
        chat_history = RedisChatMessageHistory(
            session_id=request.session_id,
            url=REDIS_URL
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=chat_history,
            return_messages=True
        )

        # Retrieve RAG doc (if uploaded)
        doc_context = redis_client.get(f"doc:{request.session_id}")
        if doc_context:
            doc_context = doc_context.decode()
            request.query = f"Using the following uploaded document as context, answer the question:\n\n{doc_context[:3000]}\n\n{request.query}"

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            agent_kwargs={"prompt": prompt}
        )

        response = agent.invoke({"input": request.query})
        content = response.get("output") if isinstance(response, dict) else str(response)

        # Retry if vague
        if len(content.split()) < 100 or "i'm not sure" in content.lower() or "source" not in content.lower():
            followup = (
                f"Rewrite the answer with deeper analysis, comparisons, and cited sources if possible:\n\n{request.query}"
            )
            retry_response = agent.invoke({"input": followup})
            content = retry_response.get("output") if isinstance(retry_response, dict) else str(retry_response)

        return JSONResponse(content={"response": content})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return PlainTextResponse(str(e), status_code=500)

@app.post("/report")
async def generate_report(request: QueryRequest):
    try:
        chat_history = RedisChatMessageHistory(
            session_id=request.session_id,
            url=REDIS_URL
        )
        messages = chat_history.messages

        if not messages:
            return JSONResponse(content={"error": "No history found for this session."}, status_code=404)

        user_queries = [msg.content for msg in messages if msg.type == "human"]
        joined_queries = "\n".join(f"- {q}" for q in user_queries)

        report_prompt = (
            "You are an expert researcher. Based on the following user queries, generate a personalized, multi-section report. "
            "Include intro, key areas of interest, deep dives, comparisons, and cited sources. \n\n"
            f"User's questions:\n{joined_queries}"
        )

        report_response = llm.invoke(report_prompt)
        report_content = report_response.content.strip() if hasattr(report_response, "content") else str(report_response)

        return JSONResponse(content={"report": report_content})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return PlainTextResponse(str(e), status_code=500)

