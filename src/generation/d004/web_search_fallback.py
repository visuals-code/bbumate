from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import requests
from typing import Optional, Dict


# 벡터 DB에서 관련 문서를 찾지 못했을 때 웹 검색을 수행하는 클래스
class WebSearchFallback:
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        search_api_key: str = None,
    ):
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self.model = model or os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")
        self.search_api_key = search_api_key or os.getenv("SEARCH_API_KEY")
        self.llm = ChatUpstage(api_key=self.api_key, model=self.model)

        # 웹 검색 결과를 답변으로 변환
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 웹 검색 결과를 바탕으로 질문에 답변하는 AI입니다.\n"
                    "검색 결과를 참고하여 신혼부부 지원정책 및 혜택에 대해 정확하고 간결하게 답변하세요.\n"
                    "검색 결과가 관련이 없다면 '관련 정보를 찾을 수 없습니다'라고 답변하세요.\n\n"
                    "검색 결과:\n{search_results}",
                ),
                ("human", "질문: {question}"),
            ]
        )

        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()

    def search_web(self, query: str, num_results: int = 3) -> Optional[Dict]:

        # Tavily API 사용

        if not self.search_api_key:
            print("[경고] 검색 API 키가 설정되지 않았습니다.")
            return None

        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.search_api_key,
                "query": query,
                "max_results": num_results,
                "search_depth": "basic",
                "include_answer": True,
            }

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = []
                raw_results = []

                if "results" in data:
                    for result in data["results"][:num_results]:
                        title = result.get("title", "")
                        content = result.get("content", "")
                        url = result.get("url", "")

                        results.append({"title": title, "url": url, "content": content})
                        raw_results.append(
                            f"제목: {title}\n내용: {content}\n출처: {url}\n"
                        )

                return {
                    "raw_text": "\n\n".join(raw_results) if raw_results else None,
                    "results": results,
                }
            else:
                print(f"[오류] 웹 검색 실패: {response.status_code}")
                return None

        except Exception as e:
            print(f"[오류] 웹 검색 중 오류 발생: {e}")
            return None

    def search_and_answer(self, question: str) -> Dict[str, any]:

        print(f"[웹 검색] 질문: {question}")

        # 웹 검색 수행
        search_results = self.search_web(question)

        if not search_results:
            return {
                "answer": "죄송합니다. 웹 검색 결과를 가져올 수 없습니다. "
                "데이터베이스에도 관련 정보가 없어 답변을 드릴 수 없습니다.",
                "search_results": None,
                "source": "web_search_failed",
            }

        # 검색 결과를 바탕으로 답변 생성
        answer = self.answer_chain.invoke(
            {"question": question, "search_results": search_results["raw_text"]}
        )

        return {
            "answer": answer,
            "search_results": search_results,
            "results": search_results.get("results", []),
            "source": "web_search",
        }

    def format_fallback_response(self, question: str) -> str:
        return (
            f"죄송합니다. '{question}'에 대한 정보를 데이터베이스에서 찾을 수 없습니다.\n"
            "다른 방식으로 질문을 다시 작성해 주시거나, "
            "관련 기관에 직접 문의해 보시는 것을 권장드립니다."
        )
