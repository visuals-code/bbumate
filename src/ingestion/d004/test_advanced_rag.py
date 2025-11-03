import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from advanced_rag_chain import AdvancedRAGChain

# .env 파일 로드
load_dotenv()


def print_result(result: dict):
    """결과를 보기 좋게 출력"""
    print("\n" + "=" * 80)
    print(f"원본 질문: {result['original_question']}")
    if result["final_question"] != result["original_question"]:
        print(f"최종 질문: {result['final_question']}")
    print(f"라우팅 상태: {result['routing_status']}")
    print(f"검색된 문서: {result['documents_retrieved']}개")
    print(f"관련 문서: {result['relevant_documents']}개")
    print(f"재작성 횟수: {result['rewrite_count']}회")
    print(f"출처: {result['source']}")
    print(f"\n답변:\n{result['answer']}")
    print("=" * 80)


def run_advanced_test():

    print("RAG 파이프라인 테스트 시작")

    # RAG 체인 초기화
    try:
        rag = AdvancedRAGChain(max_rewrite_attempts=2)
    except Exception as e:
        print(f" RAG 체인 초기화 실패: {e}")
        return

    # 테스트 케이스들
    test_cases = [
        {"name": "명확한 질문", "question": "신혼부부 백화점 혜택 알려줘"},
        {"name": "모호한 질문 (재질문 필요)", "question": "혜택"},
        {"name": "짧은 질문 (재작성 필요 가능)", "question": "백화점"},
        {"name": "구체적인 질문", "question": "신혼부부 주택 구입 지원 정책"},
        {
            "name": "DB에 없을 가능성이 높은 질문 (웹 검색 필요)",
            "question": "2025년 최신 신혼부부 정책",
        },
    ]

    results = []

    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'='*80}")
        print(f"테스트 케이스 {idx}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*80}")

        try:
            result = rag.invoke(test_case["question"])
            results.append({"test_case": test_case["name"], "result": result})
            print_result(result)

        except Exception as e:
            print(f"\n 오류 발생: {e}")
            import traceback

            traceback.print_exc()

    # 전체 요약
    print("테스트 요약")

    for idx, item in enumerate(results, 1):
        result = item["result"]
        print(f"\n{idx}. {item['test_case']}")
        print(f"   - 라우팅: {result['routing_status']}")
        print(f"   - 재작성: {result['rewrite_count']}회")
        print(f"   - 출처: {result['source']}")
        print(
            f"   - 관련 문서: {result['relevant_documents']}/{result['documents_retrieved']}"
        )

    print(f"테스트 완료! {len(results)}/{len(test_cases)}개 실행됨")


if __name__ == "__main__":
    run_advanced_test()
