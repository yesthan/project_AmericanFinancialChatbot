import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .financial_advisor import FinancialAdvisor
from django.shortcuts import render  # HTML 페이지를 반환하기 위해 필요

# 로거 설정
logger = logging.getLogger(__name__)

@csrf_exempt
def chatbot(request):
    # 디버깅용 로그 (로그 레벨을 INFO로 설정)
    logger.info("Request method: %s", request.method)
    logger.info("Request body: %s", request.body.decode('utf-8'))
    logger.info("Headers: %s", request.headers)
    
    if request.method == 'POST':
        logger.info("Request Meta: %s", request.META)  # 디버그용, 요청 메타 정보 확인
        try:
            # 요청 본문을 JSON으로 파싱
            data = json.loads(request.body)
            logger.info(f"Received data: {data}")  # 요청 데이터를 로깅 (INFO level)

            query = data.get('query', '')  # query 값을 가져옵니다.
            
            # FinancialAdvisor 객체 생성 후, query를 처리합니다.
            advisor = FinancialAdvisor()
            response = advisor.process_query(query)

            # 처리된 결과를 JSON 응답으로 반환
            return JsonResponse({'response': response})

        except json.JSONDecodeError:
            # JSON 디코딩 오류 처리
            logger.error("JSON decoding error: Invalid JSON format.")  # 오류 로깅
            return JsonResponse({'error': '잘못된 JSON 형식입니다.'}, status=400)
        except Exception as e:
            # 다른 예외 처리
            logger.error(f"Error occurred: {str(e)}")  # 오류 로깅
            return JsonResponse({'error': str(e)}, status=500)
    else:
        # POST 이외의 요청 처리
        logger.warning("Invalid request method. Only POST requests are supported.")  # 경고 로그
        return JsonResponse({'error': 'POST 요청만 지원합니다.'}, status=405)

# 새로운 index 뷰 추가 (홈페이지)
def index(request):
    return render(request, 'index.html')  # index.html 파일을 반환