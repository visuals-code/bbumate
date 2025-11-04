# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__name__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "chains" / "d004"))
sys.path.insert(0, str(project_root / "src" / "generation" / "d004"))
sys.path.insert(0, str(project_root / "src" / "retrieval" / "d004"))

from chain import AdvancedRAGChain

# .env 파일 로드
load_dotenv()


# test.py에 추가하여 DB에 데이터가 있는지 확인
def check_vectorstore():
    from retrieval import load_retriever

    retriever = load_retriever()

    # 전체 문서 개수 확인
    collection = retriever.vectorstore.get()
    print(f"총 문서 개수: {len(collection['ids'])}")

    # 샘플 검색
    docs = retriever.invoke("신혼부부")
    print(f"검색 결과: {len(docs)}개")
    for doc in docs[:3]:
        print(f"- {doc.page_content[:100]}...")


def print_separator(char="=", length=80):
    """구분선 출력"""
    print("\n" + char * length)


def print_result(result: dict):
    """RAG 결과를 보기 좋게 출력"""
    print_separator()
    print(f"📝 원본 질문: {result['original_question']}")

    if result["final_question"] != result["original_question"]:
        print(f"🔄 최종 질문: {result['final_question']}")

    print(f"🎯 라우팅 상태: {result['routing_status']}")
    print(f"📚 검색된 문서: {result['documents_retrieved']}개")
    print(f"✅ 관련 문서: {result['relevant_documents']}개")
    print(f"🔁 재작성 횟수: {result['rewrite_count']}회")
    print(f"📍 출처: {result['source']}")

    print(f"\n💬 답변:\n{result['answer']}")

    # 출처 정보 출력
    if result["sources"]:
        print(f"\n📎 출처 정보 ({len(result['sources'])}개):")
        for idx, source in enumerate(result["sources"], 1):
            print(f"  [{idx}] {source['title']}")
            if source["url"]:
                print(f"      🔗 {source['url']}")
            print(f"      📄 {Path(source['source']).name}")
    else:
        print("\n📎 출처: 없음")

    print_separator()


def run_basic_tests():
    """기본 테스트 케이스 실행"""
    print("\n" + "=" * 80)
    print("🚀 RAG 파이프라인 기본 테스트 시작")
    print("=" * 80)

    # RAG 체인 초기화
    try:
        rag = AdvancedRAGChain(max_rewrite_attempts=1)
        print("✅ RAG 체인 초기화 성공")
    except Exception as e:
        print(f"❌ RAG 체인 초기화 실패: {e}")
        return

    # 기본 테스트 케이스
    test_cases = [
        {
            "name": "벡터 DB 검색 (정상)",
            "question": "신혼부부 백화점 혜택 알려줘",
            "region": None,
            "housing_type": None,
            "expected": "vectorstore",
        },
        {
            "name": "모호한 질문 (재질문 필요)",
            "question": "혜택",
            "region": None,
            "housing_type": None,
            "expected": "clarification",
        },
        {
            "name": "시간 표현 (웹 검색)",
            "question": "오늘 발표된 신혼부부 정책",
            "region": None,
            "housing_type": None,
            "expected": "web_search",
        },
        {
            "name": "특정 연도 (웹 검색)",
            "question": "2025년 11월 신혼부부 대출 금리",
            "region": None,
            "housing_type": None,
            "expected": "web_search",
        },
        {
            "name": "희귀 주제 (웹 검색)",
            "question": "독도 거주 신혼부부 혜택",
            "region": None,
            "housing_type": None,
            "expected": "web_search",
        },
    ]

    results = []

    for idx, test_case in enumerate(test_cases, 1):
        print_separator("-")
        print(f"테스트 {idx}/{len(test_cases)}: {test_case['name']}")
        print_separator("-")

        try:
            result = rag.invoke(
                question=test_case["question"],
                region=test_case["region"],
                housing_type=test_case["housing_type"],
            )
            results.append({"test_case": test_case, "result": result, "success": True})
            print_result(result)

            # 기대값 검증
            if test_case["expected"]:
                actual_source = result.get("source", "unknown")
                expected = test_case["expected"]

                if actual_source == expected:
                    print(f"✅ 검증 통과: {expected} 출처로 답변됨")
                else:
                    print(f"⚠️  검증 실패: 예상 {expected}, 실제 {actual_source}")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "test_case": test_case,
                    "result": None,
                    "success": False,
                    "error": str(e),
                }
            )

    # 전체 요약
    print_summary(results)


def run_custom_test(question: str, region: str = None, housing_type: str = None):
    """사용자 정의 질문 테스트"""
    print("\n" + "=" * 80)
    print("🔍 사용자 정의 질문 테스트")
    print("=" * 80)

    try:
        rag = AdvancedRAGChain(max_rewrite_attempts=1)
        print("✅ RAG 체인 초기화 성공\n")
    except Exception as e:
        print(f"❌ RAG 체인 초기화 실패: {e}")
        return

    try:
        result = rag.invoke(question=question, region=region, housing_type=housing_type)
        print_result(result)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


def print_summary(results: list):
    """테스트 결과 요약 출력"""
    print("\n" + "=" * 80)
    print("📊 테스트 결과 요약")
    print("=" * 80)

    total = len(results)
    success = sum(1 for r in results if r["success"])
    failed = total - success

    print(f"\n✅ 성공: {success}/{total}")
    print(f"❌ 실패: {failed}/{total}")

    # 상세 결과
    print("\n상세 결과:")
    for idx, item in enumerate(results, 1):
        test_case = item["test_case"]

        if item["success"]:
            result = item["result"]
            print(f"\n{idx}. ✅ {test_case['name']}")
            print(f"   질문: {test_case['question']}")
            print(f"   라우팅: {result['routing_status']}")
            print(f"   재작성: {result['rewrite_count']}회")
            print(f"   출처: {result['source']}")
            print(
                f"   관련 문서: {result['relevant_documents']}/{result['documents_retrieved']}"
            )
        else:
            print(f"\n{idx}. ❌ {test_case['name']}")
            print(f"   질문: {test_case['question']}")
            print(f"   오류: {item.get('error', 'Unknown error')}")

    print_separator()


def interactive_test():
    """대화형 테스트 모드"""
    print("\n" + "=" * 80)
    print("💬 대화형 테스트 모드")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요")
    print("=" * 80)

    try:
        rag = AdvancedRAGChain(max_rewrite_attempts=1)
        print("✅ RAG 체인 초기화 성공\n")
    except Exception as e:
        print(f"❌ RAG 체인 초기화 실패: {e}")
        return

    while True:
        try:
            question = input("\n❓ 질문을 입력하세요: ").strip()

            if question.lower() in ["quit", "exit", "종료", "나가기"]:
                print("👋 테스트를 종료합니다.")
                break

            if not question:
                continue

            result = rag.invoke(question=question)
            print_result(result)

        except KeyboardInterrupt:
            print("\n\n👋 테스트를 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 파이프라인 테스트")
    parser.add_argument(
        "--mode",
        choices=["basic", "custom", "interactive"],
        default="basic",
        help="테스트 모드 선택 (basic: 기본 테스트, custom: 사용자 정의, interactive: 대화형)",
    )
    parser.add_argument(
        "--question", type=str, help="사용자 정의 질문 (custom 모드에서 사용)"
    )
    parser.add_argument("--region", type=str, help="거주지역 필터")
    parser.add_argument("--housing-type", type=str, help="주거형태 필터")

    args = parser.parse_args()

    if args.mode == "basic":
        run_basic_tests()
    elif args.mode == "custom":
        if not args.question:
            print("❌ custom 모드에서는 --question이 필요합니다")
            parser.print_help()
        else:
            run_custom_test(args.question, args.region, args.housing_type)
    elif args.mode == "interactive":
        interactive_test()
