from asyncio.log import logger
import logging
from groq import APIError
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import asyncio
import json
import time
from pydantic import BaseModel
from typing import List, Any
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai import LLMConfig
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()
groq_apiKey = os.getenv("GROQ_API_KEY")
gemini_apiKey = os.getenv("GEMINI_API_KEY")

# Initialize LLM
llm = ChatGroq(
    api_key=groq_apiKey,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    max_tokens=None,
    timeout=None,
)

# DuckDuckGo Search tool
def search(query: str) -> Any:
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=10)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    return search.invoke(query)

# Define data models
class Course_Data(BaseModel):
    module: str
    concepts: List[str]  # List of concepts for a single module

class Course_DataResponse(BaseModel):
    course_data: List[Course_Data]

# LLM Configuration for extraction
gemini_config = LLMConfig(
    provider="gemini/gemini-2.0-flash", 
    api_token="env:GEMINI_API_TOKEN"
)

# Modified crawler function to handle each website separately
async def crawl_single_website(website: str) -> Any:
    # 1. Define the LLM extraction strategy
    llm_strategy = LLMExtractionStrategy(
        llm_config=gemini_config,
        schema=Course_DataResponse.model_json_schema(),
        extraction_type="schema",
        instruction="Extract all concepts and categorize them into digestible modules. Each module should include a list of related concepts.",
        chunk_token_threshold=4000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
    )

    # 2. Build the crawler config
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS
    )

    # 3. Create browser config
    browser_cfg = BrowserConfig(headless=True)

    try:
        # Use the context manager pattern instead of explicit start/stop
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            # 4. Crawl the provided website
            result = await crawler.arun(
                url=website,
                config=crawl_config
            )
            
            if result.success:
                try:
                    # Convert the string output into a Python data structure
                    data = json.loads(result.extracted_content)
                    return data
                except json.JSONDecodeError as e:
                    return {"error": f"Failed to parse LLM output as JSON: {e}"}
                except Exception as e:
                    return {"error": f"An error occurred while processing the data: {e}"}
            else:
                return {"error": result.error_message}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Retry decorated function that runs each crawl in isolation
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=before_sleep_log(logger, logging.INFO),
    retry=retry_if_exception_type((APIError, Exception))  # Handle more exception types
)
def crawl_website(website: str) -> Any:
    # Add a delay to ensure previous processes are closed
    time.sleep(2)
    # Run in completely isolated context each time
    try:
        return asyncio.run(crawl_single_website(website))
    except RuntimeError as e:
        # Handle "Event loop is already running" error
        if "Event loop is already running" in str(e):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(crawl_single_website(website))
        else:
            raise

# Define tools
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search,
        description="REQUIRED FIRST STEP: Search for 3-5 relevant URLs about the topic. Always use this tool first to find source material."
    ),
    Tool(
        name="Crawl Website",
        func=crawl_website,
        description="REQUIRED SECOND STEP: Extract detailed content from URLs found by DuckDuckGo. Use this on at least 2-3 different sources."
    ),
]

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a research assistant with access to tools. Depending on the user’s query, you may:
- Use DuckDuckGo Search to gather initial info
- Crawl webpages to dig deeper
- Or proceed if you already know the answer

Use your judgment to determine which tool to use and when.
"""
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent execution function with retries
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(APIError)
)
def execute_groq_agent(prompt_text):
    answer = agent_executor.invoke({"input": prompt_text})
    print(answer['output'])
    return answer

# Create the agent and executor
agent = create_tool_calling_agent(llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    max_iterations=200,
    return_intermediate_steps=True
)

# Example usage
if __name__ == "__main__":
    prompt_text = "Generate me a full course on Web Development."
    execute_groq_agent(prompt_text)