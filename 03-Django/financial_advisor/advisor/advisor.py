import numpy as np
import pandas as pd
import torch
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from difflib import get_close_matches
from .advisor import FinancialAdvisor

class Exchange:
    def __init__(self):
        pass
    
class NewsCollector:
    def __init__(self):
        pass
    
class StockTools:
    def __init__(self):
        pass