import requests
from bs4 import BeautifulSoup

class Exchange:
    def __init__(self):
        self.base_url = "https://finance.naver.com/marketindex/"
    
    def get_usd_exchange_rate(self) -> dict:
        """네이버 증권에서 미국 USD 환율 정보를 가져옵니다."""
        try:
            response = requests.get(self.base_url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()  # 요청이 성공했는지 확인
            soup = BeautifulSoup(response.text, "html.parser")
            
            # 미국 USD 환율 데이터 가져오기
            usd_data = soup.select_one("div.market1 ul.data_lst li.on")
            if not usd_data:
                return {"error": "환율 정보를 가져올 수 없습니다."}
            
            # 필요한 데이터 추출
            value = usd_data.select_one("span.value").get_text(strip=True)  # 환율 값
            change = usd_data.select_one("span.change").get_text(strip=True)  # 변동 값
            blind = usd_data.select("span.blind")[2].get_text(strip=True)  # 설명 (상승/하락)
            
            return {"value": float(value.replace(",", "")), "change": float(change.replace(",", "")), "blind": blind}
        except Exception as e:
            return {"error": f"환율 정보를 가져오는 중 오류 발생: {str(e)}"}