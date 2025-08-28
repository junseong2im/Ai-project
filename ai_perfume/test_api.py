#!/usr/bin/env python3
"""
API 테스트 스크립트
"""

import requests
import json

def test_api():
    """API 테스트"""
    print("=== Movie Scent AI API 테스트 ===\n")
    
    # 건강 상태 확인
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print("[OK] 건강 상태 확인:")
            print(f"  - 상태: {health_data.get('status', 'unknown')}")
            print(f"  - 시스템: {health_data.get('systems', {})}")
        else:
            print(f"[ERROR] 건강 상태 확인 실패: {health_response.status_code}")
    except Exception as e:
        print(f"[ERROR] 건강 상태 확인 실패: {e}")
        return
    
    # 향수 추천 테스트
    test_cases = [
        {
            "description": "로맨틱한 해변가 석양 키스신",
            "scene_type": "romance",
            "intensity": "medium"
        },
        {
            "description": "어두운 숲속 미스터리 장면",
            "scene_type": "mystery",
            "intensity": "strong"
        },
        {
            "description": "밝은 봄날 카페에서의 만남",
            "scene_type": "drama", 
            "intensity": "light"
        }
    ]
    
    print("\n=== 향수 추천 테스트 ===")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {test_case['description']}")
        
        try:
            response = requests.post(
                "http://localhost:8000/recommend_scent",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print("[SUCCESS] 성공!")
                
                # 추천 결과 확인
                recommendations = data.get('recommendations', [])
                print(f"  - 추천 개수: {len(recommendations)}")
                
                # 처리 시간
                processing_time = data.get('processing_time', 0)
                print(f"  - 처리 시간: {processing_time:.3f}초")
                
                # 시스템 모드
                system_mode = data.get('system_mode', 'UNKNOWN')
                print(f"  - 시스템 모드: {system_mode}")
                
                # 훈련된 모델 예측 확인
                if 'trained_model_predictions' in data:
                    preds = data['trained_model_predictions']
                    print("[TRAINED MODEL] 훈련된 모델 예측:")
                    print(f"    - 강도: {preds.get('intensity', 0):.1f}")
                    print(f"    - 지속시간: {preds.get('longevity_hours', 0):.1f}시간")
                    print(f"    - 확산성: {preds.get('diffusion', 0):.1f}")
                
                # 첫 번째 추천 상세 정보
                if recommendations:
                    first_rec = recommendations[0]
                    print(f"  - 첫 번째 추천: {first_rec.get('name', 'N/A')}")
                    print(f"    * 강도: {first_rec.get('intensity', 0)}")
                    print(f"    * 휘발성: {first_rec.get('volatility', 'N/A')}")
                
            else:
                print(f"[FAIL] 실패: {response.status_code}")
                print(f"응답: {response.text}")
                
        except Exception as e:
            print(f"[ERROR] 요청 실패: {e}")
    
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    test_api()