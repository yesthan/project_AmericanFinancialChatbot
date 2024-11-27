import yfinance as yf

class StockTools:
    def get_stock_price(self, tool_input: str) -> dict:
        """주식 심볼을 입력받아 현재 주가 정보를 반환합니다."""
        try:
            symbol = str(tool_input).strip().upper()
            stock = yf.Ticker(symbol)
            info = stock.info
            history = stock.history(period='1d')
            
            if history.empty:
                return {"error": f"데이터를 찾을 수 없습니다: {symbol}"}
                
            return info
        except Exception as e:
            return {"error": f"주가 조회 실패: {str(e)}"}

    def get_market_indices(self, tool_input: str = "") -> dict:
        """주요 시장 지수 정보를 반환합니다. 상승/하락 이유를 분석합니다."""
        indices = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ'}
        results = {}
        
        for symbol, name in indices.items():
            try:
                stock = yf.Ticker(symbol)
                history = stock.history(period='1d')
                if not history.empty:
                    close_price=history['Close'].iloc[-1]
                    open_price=history['Open'].iloc[0]
                    change=((close_price-open_price)/open_price)*100
                    results[name] = {
                        "price": close_price,
                        "change": change,
                        "trend":"상승" if change >=0 else "하락"
                    }
            except Exception:
                results[name] = {"error": "데이터 조회 실패"}
                
        return results