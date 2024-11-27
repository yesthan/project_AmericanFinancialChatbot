from bs4 import BeautifulSoup
import requests
from langchain.tools import tool

class NewsCollector:
    @tool("get_latest_news")
    def get_latest_news(self, tool_input: str = "") -> list:
        """야후 파이낸스에서 최신 뉴스를 가져옵니다."""
        try:
            response = requests.get(
                "https://finance.yahoo.com/news",
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            new_items=[]
            soup = BeautifulSoup(response.text, 'html.parser')
            articles1 = soup.find_all('h3', class_='clamp tw-line-clamp-3 sm:tw-line-clamp-2 yf-18q3fnf')
            articles2 = soup.find_all('h2',class_= 'tw-line-clamp-3 yf-16ne7ux')
            articles = articles1+articles2
            for article in articles:
                new_items.append(article.text)

            return new_items
        except Exception as e:
            return f"뉴스 수집 실패: {str(e)}"