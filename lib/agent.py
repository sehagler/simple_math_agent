# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:56:36 2026

@author: sehag
"""

#
from langchain_classic.agents import Tool, initialize_agent
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.chains import LLMMathChain, LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

#
class Agent_object(object):
    
    #
    def __init__(self):
        self.math_tool = None
        self.model = None
        self.tavily_tool = None
        self.wikipedia_tool = None
        self.word_problem_tool = None
        self.tools = []
        
    #
    def assemble_tools(self):
        if self.math_tool is not None:
            self.tools.append(self.math_tool)
        if self.tavily_tool is not None:
            self.tools.append(self.tavily_tool)
        if self.wikipedia_tool is not None:
            self.tools.append(self.wikipedia_tool)
        if self.word_problem_tool is not None:
            self.tools.append(self.word_problem_tool)
        
    #
    def create_math_tool(self):
        problem_chain = LLMMathChain.from_llm(llm=self.model)
        self.math_tool = \
            Tool.from_function(name="Calculator",
                               func=problem_chain.run,
                               description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.")
    
    #
    def create_model(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    #
    def create_tavily_tool(self):
        tavily_search = TavilySearchResults(max_results=1)
        self.tavily_tool = \
            Tool(name="Tavily",
                 func=tavily_search.invoke,
                 description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.")
        
    #
    def create_wikipedia_tool(self):
        wikipedia = WikipediaAPIWrapper()
        self.wikipedia_tool = \
            Tool(name="Wikipedia",
                 func=wikipedia.run,
                 description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.")
    
    #
    def create_word_problem_tool(self):
        word_problem_template = """You are a reasoning agent tasked with solving 
        the user's logic-based questions. Logically arrive at the solution, and be 
        factual. In your answers, clearly detail the steps involved and give the 
        final answer. Provide the response in bullet points. 
        Question  {question} Answer"""
        math_assistant_prompt = PromptTemplate(input_variables=["question"],
                                               template=word_problem_template
                                               )
        word_problem_chain = LLMChain(llm=self.model,
                                      prompt=math_assistant_prompt)
        self.word_problem_tool = \
            Tool.from_function(name="Reasoning Tool",
                               func=word_problem_chain.run,
                               description="Useful for when you need to answer logic-based/reasoning questions.")
    
    #
    def get_agent(self):
        agent = initialize_agent(tools=self.tools,
                                 llm=self.model,
                                 agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                 verbose=False,
                                 handle_parsing_errors=True
                                 )
        return agent