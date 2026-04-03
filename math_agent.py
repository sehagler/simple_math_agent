# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:31:58 2026

@author: sehag
"""

#
import chainlit as cl
import os

#
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "..."
os.environ["OPENAI_API_KEY"] ="..."
os.environ["TAVILY_API_KEY"] = "..."
os.environ['USER_AGENT'] = 'math_agent'

#
from lib.agent import Agent_object

@cl.on_chat_start
def math_chatbot():
    agent_object = Agent_object()
    agent_object.create_model()
    agent_object.create_math_tool()
    #agent_object.create_tavily_tool()
    agent_object.create_wikipedia_tool()
    agent_object.create_word_problem_tool()
    agent_object.assemble_tools()
    agent = agent_object.get_agent()
    cl.user_session.set("agent", agent)
    
@cl.on_message
async def process_user_query(message: cl.Message):
    agent = cl.user_session.get("agent")
    response = await agent.acall(message.content,
                                 callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(response["output"]).send()