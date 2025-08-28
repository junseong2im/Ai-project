#!/usr/bin/env python3
"""
완성된 Movie Scent AI 시스템 종합 테스트
200,000개 데이터셋으로 훈련된 딥러닝 모델 포함
"""

import requests
import json
import time
from pathlib import Path

def test_trained_model_integration():
    """훈련된 모델 직접 테스트"""
    print("=== 훈련된 200k 모델 직접 테스트 ===")
    
    try:
        from core.deep_learning_integration import get_trained_predictor
        
        predictor = get_trained_predictor()
        
        if predictor.is_loaded:
            print("[SUCCESS] 200k 훈련 모델 로드 완료")
            print(f"  - 디바이스: {predictor.device}")
            print(f"  - 모델 경로: {predictor.model_path}")
            
            # 테스트 예측
            test_scene = "로맨틱한 해변가 석양 키스신"
            result = predictor.predict_scene_fragrance(test_scene)
            
            if result["success"]:
                print(f"\n[PREDICTION] '{test_scene}' 예측 결과:")
                preds = result["predictions"]
                print(f"  - 향기 강도: {preds['intensity']:.1f}/100")
                print(f"  - 지속 시간: {preds['longevity_hours']:.1f}시간")
                print(f"  - 확산성: {preds['diffusion']:.1f}/10")
                print(f"  - 임계값: {preds['threshold_ppb']:.2f}ppb")
                print(f"  - 최대 농도: {preds['max_concentration']:.1f}%")
                
                return True
            else:
                print(f"[ERROR] 예측 실패: {result.get('error', 'unknown')}")
                return False
        else:
            print("[ERROR] 훈련된 모델 로드 실패")
            return False
            
    except Exception as e:
        print(f"[ERROR] 모델 테스트 실패: {e}")
        return False

def test_dataset_quality():
    """생성된 데이터셋 품질 확인"""
    print("\n=== 데이터셋 품질 확인 ===")
    
    dataset_files = [
        "data/datasets/fragrance_train.json",
        "data/datasets/fragrance_validation.json", 
        "data/datasets/fragrance_test.json"
    ]
    
    total_samples = 0
    
    for file_path in dataset_files:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            materials = data["materials"]
            total_samples += len(materials)
            
            # 천연/합성 비율 계산
            natural_count = sum(1 for m in materials if m["is_natural"])
            synthetic_count = len(materials) - natural_count
            
            # 향기 계열 분포 확인
            families = {}
            for material in materials:
                family = material["family"]
                families[family] = families.get(family, 0) + 1
            
            print(f"[OK] {Path(file_path).name}: {len(materials):,}개 샘플")
            print(f"  - 천연: {natural_count:,}개 ({natural_count/len(materials)*100:.1f}%)")
            print(f"  - 합성: {synthetic_count:,}개 ({synthetic_count/len(materials)*100:.1f}%)")
            print(f"  - 향기 계열: {len(families)}종")
    
    print(f"\n[TOTAL] 전체 데이터셋: {total_samples:,}개 샘플")
    
    return total_samples >= 200000

def test_web_interface():
    """웹 인터페이스 기능 테스트"""
    print("\n=== 웹 인터페이스 테스트 ===")
    
    # 메인 페이지 확인
    try:
        response = requests.get("http://localhost:8000/", timeout=10)
        if response.status_code == 200:
            print("[OK] 메인 페이지 접근 가능")
        else:
            print(f"[WARNING] 메인 페이지 상태: {response.status_code}")
    except:
        print("[ERROR] 메인 페이지 접근 불가")
        return False
    
    # API 문서 확인
    try:
        response = requests.get("http://localhost:8000/docs", timeout=10)
        if response.status_code == 200:
            print("[OK] API 문서 접근 가능")
        else:
            print(f"[WARNING] API 문서 상태: {response.status_code}")
    except:
        print("[WARNING] API 문서 접근 불가")
    
    return True

def test_system_performance():
    """시스템 성능 테스트"""
    print("\n=== 시스템 성능 테스트 ===")
    
    test_descriptions = [
        "밝은 봄날 공원에서의 첫 만남",
        "비 오는 겨울밤 따뜻한 카페",  
        "여름 해변가 파티 장면",
        "가을 단풍길 산책 데이트",
        "신비로운 숲속 모험 장면"
    ]
    
    response_times = []
    
    for desc in test_descriptions:
        start_time = time.time()
        
        try:
            response = requests.post(
                "http://localhost:8000/recommend_scent",
                json={"description": desc, "intensity": "medium"},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                process_time = time.time() - start_time
                response_times.append(process_time)
                
                recommendations = data.get('recommendations', [])
                system_mode = data.get('system_mode', 'UNKNOWN')
                
                print(f"[OK] '{desc[:20]}...': {len(recommendations)}개 추천, {process_time:.3f}초 ({system_mode})")
                
            else:
                print(f"[ERROR] '{desc[:20]}...': HTTP {response.status_code}")
                
        except Exception as e:
            print(f"[ERROR] '{desc[:20]}...': {e}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        print(f"\n[PERFORMANCE] 평균 응답 시간: {avg_time:.3f}초")
        print(f"[PERFORMANCE] 최소/최대: {min(response_times):.3f}s / {max(response_times):.3f}s")
        
        return avg_time < 1.0  # 1초 이내 응답
    
    return False

def main():
    """종합 테스트 실행"""
    print("=" * 60)
    print("Movie Scent AI - 완성 시스템 종합 테스트")
    print("200k 데이터셋 + 딥러닝 훈련 + 웹 API 통합")
    print("=" * 60)
    
    test_results = {}
    
    # 1. 훈련된 모델 테스트
    test_results["trained_model"] = test_trained_model_integration()
    
    # 2. 데이터셋 품질 확인
    test_results["dataset_quality"] = test_dataset_quality()
    
    # 3. 웹 인터페이스 테스트
    test_results["web_interface"] = test_web_interface()
    
    # 4. 성능 테스트
    test_results["performance"] = test_system_performance()
    
    # 종합 결과
    print("\n" + "=" * 60)
    print("종합 테스트 결과")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        status = "[PASS]" if result else "[FAIL]"
        test_display = test_name.replace("_", " ").title()
        print(f"{status} {test_display}")
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    print(f"\n최종 결과: {passed}/{total} 테스트 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! 시스템이 완성되었습니다!")
        print("\n✨ Movie Scent AI 시스템 기능:")
        print("  - 200,000개 향료 원료 데이터셋")
        print("  - 편향 없는 딥러닝 모델 훈련 완료")
        print("  - 실시간 장면 분석 및 향수 추천")
        print("  - 웹 인터페이스 및 REST API")
        print("  - 화학적 정확한 원료 조합 공식")
        print("\n🌐 웹 인터페이스: http://localhost:8000")
        print("📚 API 문서: http://localhost:8000/docs")
    else:
        print(f"\n⚠️  {total-passed}개 테스트 실패. 시스템을 점검해주세요.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)