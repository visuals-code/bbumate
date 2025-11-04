"""웹 검색 모듈 (Web Search).

외부 정보가 필요할 때 웹을 검색하여 정보를 수집합니다.
"""

import os
from typing import List, Union

from langchain_core.documents import Document

from src.utils.d001.logger import get_logger

logger = get_logger(__name__)


class WebSearchTool:
    """웹 검색 도구.

    Tavily API를 사용하여 웹 검색을 수행합니다.
    """

    def __init__(self, max_results: int = 3) -> None:
        """WebSearchTool을 초기화합니다.

        Args:
            max_results: 최대 검색 결과 수.
        """
        self.max_results = max_results
        self._init_search_tool()

    def _init_search_tool(self) -> None:
        """검색 도구를 초기화합니다.

        Returns:
            None. Tavily 검색 도구를 초기화합니다.
        """
        try:
            # Tavily API 키 확인
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                logger.warning("TAVILY_API_KEY not found. Please set it in .env file.")
                self.search_tool = None
                return

            # Tavily 클라이언트 초기화
            from tavily import TavilyClient

            self.search_tool = TavilyClient(api_key=api_key)
            logger.info("Initialized Tavily search tool successfully")

        except ImportError:
            logger.warning(
                "Tavily not available. Install: pip install tavily-python"
            )
            self.search_tool = None
        except Exception as e:
            logger.error("Failed to initialize Tavily search tool: %s", e)
            self.search_tool = None

    def search(self, query: str) -> List[Document]:
        """웹 검색을 수행합니다 (Tavily).

        Args:
            query: 검색 쿼리.

        Returns:
            검색 결과 문서 리스트. 실패 시 빈 리스트 반환.
        """
        if self.search_tool is None:
            logger.warning("Web search tool not initialized")
            return []

        logger.info("Searching web with Tavily for: %s", query)

        try:
            # Tavily 검색 실행
            response = self.search_tool.search(
                query=query,
                max_results=self.max_results,
                search_depth="basic",  # "basic" 또는 "advanced"
                include_domains=[],
                exclude_domains=[]
            )

            docs = []
            # 결과 파싱
            if "results" in response and isinstance(response["results"], list):
                for result in response["results"]:
                    title = result.get("title", "Web Search Result")
                    content = result.get("content", "")
                    url = result.get("url", "unknown")

                    docs.append(
                        Document(
                            page_content=f"{title}\n\n{content}",
                            metadata={
                                "source": url,
                                "title": title,
                                "url": url,
                                "query": query,
                                "search_engine": "tavily",
                                "score": result.get("score", 0.0)
                            }
                        )
                    )

            logger.info("Found %d web search results from Tavily", len(docs))
            return docs

        except Exception as e:  # pylint: disable=broad-except
            # 폴백: 웹 검색 실패 시 빈 리스트 반환하여 처리 계속 진행
            logger.error("Tavily web search failed: %s", e)
            return []
            
    async def asearch(self, query: str) -> List[Document]:
        """비동기적으로 웹 검색을 수행합니다 (Tavily).

        Args:
            query: 검색 쿼리.

        Returns:
            검색 결과 문서 리스트. 실패 시 빈 리스트 반환.
        """
        if self.search_tool is None:
            logger.warning("Web search tool not initialized")
            return []

        logger.info("Async searching web with Tavily for: %s", query)

        try:
            # Tavily는 비동기 메서드를 제공하지 않으므로 동기 메서드 사용
            # asyncio.to_thread를 사용하여 블로킹 호출을 비동기로 실행
            import asyncio

            response = await asyncio.to_thread(
                self.search_tool.search,
                query=query,
                max_results=self.max_results,
                search_depth="basic",
                include_domains=[],
                exclude_domains=[]
            )

            docs = []
            # 결과 파싱
            if "results" in response and isinstance(response["results"], list):
                for result in response["results"]:
                    title = result.get("title", "Web Search Result")
                    content = result.get("content", "")
                    url = result.get("url", "unknown")

                    docs.append(
                        Document(
                            page_content=f"{title}\n\n{content}",
                            metadata={
                                "source": url,
                                "title": title,
                                "url": url,
                                "query": query,
                                "search_engine": "tavily",
                                "score": result.get("score", 0.0)
                            }
                        )
                    )

            logger.info("Found %d async web search results from Tavily", len(docs))
            return docs

        except Exception as e:  # pylint: disable=broad-except
            # 폴백: 웹 검색 실패 시 빈 리스트 반환하여 처리 계속 진행
            logger.error("Tavily async web search failed: %s", e)
            return []

    def search_and_format(self, query: str) -> str:
        """웹 검색 후 결과를 포맷팅된 문자열로 반환합니다.

        Args:
            query: 검색 쿼리.

        Returns:
            포맷팅된 검색 결과 문자열.
        """
        docs = self.search(query)

        if not docs:
            return "웹 검색 결과를 찾을 수 없습니다."

        formatted_results = []
        for i, doc in enumerate(docs, 1):
            formatted_results.append(
                f"검색 결과 {i}:\n{doc.page_content}\n"
            )

        return "\n".join(formatted_results)


class MockWebSearchTool:
    """웹 검색 Mock 도구 (테스트/개발용).

    실제 웹 검색 대신 Mock 데이터를 반환합니다.
    """

    def __init__(self, max_results: int = 3) -> None:
        """MockWebSearchTool을 초기화합니다.

        Args:
            max_results: 최대 검색 결과 수.
        """
        self.max_results = max_results

    def search(self, query: str) -> List[Document]:
        """Mock 검색을 수행합니다.

        Args:
            query: 검색 쿼리.

        Returns:
            Mock 검색 결과 문서 리스트.
        """
        logger.info("[MOCK] Searching web for: %s", query)

        # Mock 결과 반환 - 더 구체적인 내용으로
        mock_results = [
            Document(
                page_content=f"""
[웹 검색 결과 1]
'{query}'에 대한 최신 정보입니다.

해당 지역의 신혼부부 주택지원정책은 다음과 같습니다:
- 지원 대상: 혼인신고일로부터 7년 이내 신혼부부
- 지원 내용: 전세자금 대출, 월세 지원, 주택 우선 공급 등
- 신청 방법: 해당 지역 주택도시공사 또는 구청 홈페이지에서 신청

자세한 내용은 해당 지역 주민센터나 주택도시공사에 문의하시기 바랍니다.
※ 위 정보는 웹에서 검색한 최신 정보입니다.
                """.strip(),
                metadata={"source": "mock_web_search", "query": query, "url": "https://example.com/policy1"}
            ),
            Document(
                page_content=f"""
[웹 검색 결과 2]
'{query}' 관련 추가 정보입니다.

신혼부부 지원정책 신청 시 필요서류:
1. 혼인관계증명서
2. 주민등록등본
3. 소득증명서류
4. 무주택 확인서

온라인 신청이 가능하며, 자세한 사항은 해당 지역 자치단체 홈페이지를 참고하세요.
※ 위 정보는 웹에서 검색한 최신 정보입니다.
                """.strip(),
                metadata={"source": "mock_web_search", "query": query, "url": "https://example.com/policy2"}
            ),
        ]

        return mock_results[:self.max_results]
        
    async def asearch(self, query: str) -> List[Document]:
        """비동기적으로 Mock 검색을 수행합니다.

        Args:
            query: 검색 쿼리.

        Returns:
            Mock 검색 결과 문서 리스트.
        """
        return self.search(query)

    def search_and_format(self, query: str) -> str:
        """Mock 검색 후 포맷팅합니다.

        Args:
            query: 검색 쿼리.

        Returns:
            포맷팅된 검색 결과 문자열.
        """
        docs = self.search(query)
        return "\n\n".join([doc.page_content for doc in docs])


def create_web_search_tool(
    max_results: int = 3, use_mock: bool = False
) -> Union[WebSearchTool, MockWebSearchTool]:
    """웹 검색 도구 생성 헬퍼 함수.

    Args:
        max_results: 최대 검색 결과 수.
        use_mock: Mock 도구 사용 여부 (개발/테스트용).

    Returns:
        웹 검색 도구 인스턴스.
    """
    if use_mock:
        logger.info("Using mock web search tool")
        return MockWebSearchTool(max_results=max_results)
    else:
        return WebSearchTool(max_results=max_results)
