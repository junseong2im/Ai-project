#!/usr/bin/env python3
"""
영화용 캡슐 방향제 시스템 테스트
감독 요구사항에 맞는 정확한 제조 공식 생성
"""

import requests
import json

def test_capsule_api():
    """캡슐 API 테스트"""
    print("=" * 70)
    print("[MOVIE] 영화용 캡슐 방향제 제조 시스템 테스트")
    print("=" * 70)
    
    # 테스트 시나리오들
    test_scenarios = [
        {
            "name": "로맨틱 키스신",
            "scene_description": "석양이 지는 해변가에서 두 연인이 포옹하며 키스하는 로맨틱한 장면",
            "target_duration": 8.0
        },
        {
            "name": "액션 추격신",
            "scene_description": "어두운 골목길에서 벌어지는 긴장감 넘치는 추격 액션 장면",
            "target_duration": 5.0
        },
        {
            "name": "슬픈 이별신",
            "scene_description": "비 오는 역에서 헤어지는 커플의 슬픈 이별 장면",
            "target_duration": 10.0
        },
        {
            "name": "평화로운 자연",
            "scene_description": "맑은 봄날 꽃밭에서 명상하는 평화로운 장면",
            "target_duration": 6.0
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n[테스트 {i}] {scenario['name']}")
        print("-" * 50)
        
        try:
            response = requests.post(
                "http://localhost:8000/api/movie_capsule",
                json={
                    "scene_description": scenario["scene_description"],
                    "target_duration": scenario["target_duration"]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print("[성공] 캡슐 제조 공식 생성 완료!")
                
                # 기본 정보
                print(f"• 장면: {scenario['scene_description']}")
                print(f"• 목표 지속시간: {data['target_duration_seconds']}초")
                print(f"• 확산 제어: {data['diffusion_control']*100:.1f}% (낮은 확산)")
                print(f"• 예상 비용: ${data['estimated_cost_per_unit']:.4f}/개")
                
                # 원료 구성
                print(f"\n[MATERIALS] 원료 구성 (총 {len(data['raw_materials'])}종):")
                for material in data["raw_materials"]:
                    print(f"  - {material['name'].replace('_', ' ').title()}: {material['amount_ml']:.3f}ml ({material['percentage']:.1f}%)")
                    print(f"    └ 기능: {material['function']}")
                
                # 제조 순서 (처음 3단계만)
                print(f"\n[PROCESS] 제조 순서 (처음 3단계):")
                for j, step in enumerate(data["production_sequence"][:3], 1):
                    print(f"  {j}. {step}")
                
                # 캡슐 사양
                print(f"\n[CAPSULE] 캡슐 사양:")
                print(f"  - 방식: {data['encapsulation_method']}")
                print(f"  - 활성화: {data['activation_mechanism']}")
                
                print(f"\n처리 시간: {data['processing_time']:.3f}초")
                print(f"ML 강화: {'예' if data['ml_enhanced'] else '아니오'}")
                
            else:
                print(f"[실패] HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"[오류] {e}")
        
        print()

def test_direct_formulation():
    """직접 제조 시스템 테스트"""
    print("=" * 70)
    print("[LAB] 직접 제조 시스템 테스트")  
    print("=" * 70)
    
    try:
        from core.movie_capsule_formulator import get_capsule_formulator
        
        formulator = get_capsule_formulator()
        
        # 테스트 장면
        scene = "미스터리한 숲속에서 벌어지는 긴장감 넘치는 장면"
        duration = 7.0
        
        print(f"장면: {scene}")
        print(f"목표 지속시간: {duration}초")
        
        # 공식 생성
        formula = formulator.formulate_capsule(scene, duration)
        
        # 상세 보고서 출력
        report = formulator.generate_detailed_report(formula)
        print(report)
        
        print("[성공] 직접 제조 시스템 정상 동작!")
        
    except Exception as e:
        print(f"[오류] 직접 제조 실패: {e}")

def main():
    print("Movie Scent AI - 캡슐 방향제 제조 시스템")
    print("감독 요구사항: 장면 → 향 → 3-10초 지속 → 낮은 확산 → 정확한 공식\n")
    
    # 1. API 테스트
    test_capsule_api()
    
    # 2. 직접 제조 테스트  
    test_direct_formulation()
    
    print("=" * 70)
    print("[COMPLETE] 모든 테스트 완료!")
    print("감독님이 원하시는 영화용 캡슐 방향제 제조 시스템이 준비되었습니다!")
    print("=" * 70)

if __name__ == "__main__":
    main()