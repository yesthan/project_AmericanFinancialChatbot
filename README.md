# project_AmericanFinancialChatbot
미국 주식 초보를 위한 미국 주식 정보 챗봇 미정이 프로젝트입니다.
[미정이 프로젝트 Full Story](https://velog.io/@jiiiw/미국-주식-초보를-위한-주식-정보-제공-챗봇-프로젝트)

# 0. 가상 환경 생성
- Folder : **00-Conda**
- File name : **requirements.txt**

[Anaconda 환경 없을 시 다운로드](https://www.anaconda.com/download)

- anaconda prompt 실행

```
conda create -n [가상환경명] python=3.10
conda activate [가상환경명]
pip install -r requirements.txt		# 같은 디렉토리 내에 존재해야 함
```
----------------------------------------------------------------------
# 1. 데이터 수집
- Folder : **01-Crawling**
- File name : **NHterms_Crawling.ipynb**
----------------------------------------------------------------------
# 2. 모델 다운로드
- Folder : **02-Model**
- Model download : [Model download](https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B)
- Model id : **MLP-KTLim/llama-3-Korean-Bllossom-8B**
- File name : **chat.ipynb**
----------------------------------------------------------------------
# 3. Prototype
- Folder : **03-Django**
- Model id : **Bllossom/llama-3.2-Korean-Bllossom-3B**
- 실행 전 financial_advisor/settings.py IP 주소를 실행하고자 하는 디바이스의 IP 주소로 반드시 변경

- anaconda prompt 실행
```conda activate [가상환경명]	# 이미 활성화돼 있다면 무시
python manage.py runserver localhost:8000	# 혹은 localhost:8080
```
----------------------------------------------------------------------
# LICENSE
이 프로젝트는 META LLAMA 3 COMMUNITY, Apache-2.0 라이선스를 따릅니다. 자세한 내용은 [META LLAMA 3 COMMUNITY](https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/LICENSE), [LICENSE](LICENSE) 파일을 참고해 주세요.
