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
from .newscollector import NewsCollector
from .stocktools import StockTools
from .exchange import Exchange

class FinancialAdvisor:
    def __init__(self):
        # 기본 LLM 설정
        self.gpt = ChatOpenAI(
            model_name="gpt-4", 
            temperature=0.7,
            api_key="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"
        )
        # Bllossom 모델 설정
        self.setup_bllossom()
        
        # 도구 설정
        self.news_collector = NewsCollector()
        self.stock_tools = StockTools()
        
        # 메모리 설정
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
        
        # 도구 목록 설정
        self.tools = [
            Tool(
                name="News",
                func=self.news_collector.get_latest_news,
                description="최신 금융 뉴스를 가져옵니다"
            ),
            Tool(
                name="StockPrice",
                func=self.stock_tools.get_stock_price,
                description="특정 주식의 현재 가격 정보를 가져옵니다"
            ),
            Tool(
                name="MarketIndices",
                func=self.stock_tools.get_market_indices,
                description="주요 시장 지수 정보를 가져옵니다"
            )
        ]
        
        # 체인 설정
        self.setup_chains()

    def setup_bllossom(self):
        model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        self.bllossom = HuggingFacePipeline(pipeline=pipe)

    def setup_chains(self):
        # 의도 분석 체인
        intent_prompt = PromptTemplate(
            input_variables=["query"],
            template="""다음 질문의 의도를 파악해서 가장 적절한 번호 하나만 알려주세요.

                        질문: {query}

                        1번: 주식이나 투자에 대한 일반적인 설명이 필요한 경우
                        2번: 특정 주식의 현재 가격을 알고 싶은 경우
                        3번: 전체 시장 상황이나 동향을 알고 싶은 경우
                        4번: 최신 뉴스나 소식을 알고 싶은 경우
                        5번: 환율 정보를 알고 싶은 경우

                        답변:"""
        )
        
        self.intent_chain = LLMChain(
            llm=self.bllossom,
            prompt=intent_prompt,
            verbose=True
        )

        # 주식 분석 체인
        stock_prompt = PromptTemplate(
            input_variables=["stock_data","query"],
            template="""다음 주식 정보를 바탕으로 질문에 맞는 답변을 해주세요:{stock_data}
            
            질문:{query}
            답변:
            """
        )
        
        self.stock_chain = LLMChain(
            llm=self.bllossom,
            prompt=stock_prompt,
            memory=self.memory,
            verbose=True
        )
        
        # 시장 분석 체인
        market_prompt = PromptTemplate(
            input_variables=["indices", "news"],
            template="""다음 정보를 바탕으로 현재 시장 상황을 분석해서 한국말로 알려주세요:
            
            시장 지수:
            {indices}
            
            주요 뉴스:
            {news}
            """
        )
        
        self.market_chain = LLMChain(
            llm=self.gpt,
            prompt=market_prompt,
            memory=self.memory,
            verbose=True
        )

    def classify_intent(self, query: str) -> str:
        """사용자 입력의 의도를 분류합니다."""
        try:
            # 의도 분석을 위해 Bllossom 모델 사용
            response = self.intent_chain.run(query)
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                intent = numbers[0]
                if intent in ["1", "2", "3", "4"]:
                    return intent
            return "3"
            
        except Exception as e:
            print(f"의도 분류 중 오류 발생: {str(e)}")
            return "1"  # 오류 발생 시 기본값

    def process_query(self, query: str) -> str:
        try:
            # 의도 파악
            intent = self.classify_intent(query)

            if intent == "1":
                # Bllossom 모델을 사용해 질문에 대한 설명 제공
                return self.bllossom(query)

            elif intent == "2":
                # 주식 가격 조회
                symbol_prompt = PromptTemplate(
                    input_variables=["query"],
                    template="""사용자가 미국 주식과 관련된 질문을 합니다. 질문에서 미국 주식 심볼(티커)을 추출하세요. 
                                심볼은 대문자 영어로 구성된 1~5글자이며, 심볼만 반환하고 다른 설명은 포함하지 마세요. 
                                만약 질문에서 심볼을 추출할 수 없다면 미국 s&p500 심볼이라도 추출하세요.
                                질문: {query}
                                답변: 심볼(영어)
                                 """
                )
                symbol_chain = LLMChain(llm=self.gpt, prompt=symbol_prompt)
                symbol = symbol_chain.run(query).strip().upper()
                
                # Tool 실행 방식 수정
                stock_tool = [tool for tool in self.tools if tool.name == "StockPrice"][0]
                stock_data = stock_tool.run(symbol)
                
                return self.stock_chain.run(stock_data=stock_data)

            elif intent == "3":
                # 시장 동향 분석
                market_tool = [tool for tool in self.tools if tool.name == "MarketIndices"][0]
                indices = market_tool.run("")

                news_tool = [tool for tool in self.tools if tool.name == "News"][0]
                news = news_tool.run("")

                analysis_message = self.analyze_market(indices, news)

                return analysis_message

            elif intent == "4":
                # 뉴스 요약
                news_tool = [tool for tool in self.tools if tool.name == "News"][0]
                news = news_tool.run("")
                return self.gpt.predict(f"다음 뉴스를 한국말로 요약해주세요: {news}")

            elif intent == "5":
                # 환율 정보 제공
                exchange = Exchange()
                exchange_data = exchange.get_usd_exchange_rate()
                
                if "error" in exchange_data:
                    return exchange_data["error"]

                # 환율 계산 체인 실행
                exchange_prompt = PromptTemplate(
                    input_variables=['query', 'value', 'change', 'blind'],
                    template="""사용자가 환율에 대해 물어보면 현재 환율 정보를 출력해주세요.
                    만약 원화 또는 달러를 계산해달라고 요청하면 환율에 따라 결과를 계산해주세요.
                    
                    현재 환율 정보:
                    - USD 환율: {value}원
                    - 변동: {change}원 ({blind})
                    
                    질문: {query}
                    """
                )

                exchange_chain = LLMChain(llm=self.bllossom, prompt=exchange_prompt)
                response = exchange_chain.run(
                    query=query,
                    value=exchange_data["value"],
                    change=exchange_data["change"],
                    blind=exchange_data["blind"]
                )

                return response

            else:
                return "죄송합니다. 질문을 이해하지 못했습니다."
                
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"

    def analyze_market(self, indices: dict, news: list) -> str:
        """시장 지수와 뉴스를 바탕으로 상승/하락 이유를 분석합니다."""
        try:
            analysis = []
            for index, data in indices.items():
                trend = data.get("trend", "정보없음")
                price = data.get("price", "정보없음")
                change = data.get("change", "정보없음")
                related_news = self.gpt.predict(f"다음 뉴스를 한국말로 요약해주세요:{news}")

                analysis.append(
                    f"{index}({trend}: 현재 가격은 {price}, 변동률은 {change:.2f}%입니다.")
            analysis.append(related_news)

            return "\n".join(analysis)
        
        except Exception as e:
            return f"시장 분석 중 오류 발생: {str(e)}"