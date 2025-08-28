#!/usr/bin/env python3
"""
간단한 캡슐 제조 테스트
"""

def test_direct():
    """직접 테스트"""
    try:
        from core.movie_capsule_formulator import get_capsule_formulator
        
        formulator = get_capsule_formulator()
        print("[OK] 캡슐 제조기 로드 성공")
        
        scene = "로맨틱한 해변가 석양 키스신"
        duration = 7.0
        
        print(f"장면: {scene}")
        print(f"지속시간: {duration}초")
        
        formula = formulator.formulate_capsule(scene, duration)
        
        print(f"\n[결과]")
        print(f"- 총 원료 수: {len(formula.raw_materials)}종")
        print(f"- 예상 비용: ${formula.estimated_cost_per_unit:.4f}/개")
        print(f"- 확산 제어: {formula.diffusion_control}")
        
        print(f"\n[원료 구성]")
        for mat in formula.raw_materials:
            print(f"  {mat['name']}: {mat['amount_ml']:.3f}ml ({mat['percentage']:.1f}%)")
        
        print(f"\n[제조 순서 - 처음 3단계]")
        for i, step in enumerate(formula.production_sequence[:3], 1):
            print(f"  {i}. {step}")
        
        print(f"\n[캡슐 사양]")
        print(f"  포장: {formula.encapsulation_method}")
        print(f"  활성화: {formula.activation_mechanism}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Movie Scent AI - 캡슐 방향제 제조 테스트")
    print("=" * 60)
    
    success = test_direct()
    
    if success:
        print("\n[SUCCESS] 시스템이 정상 작동합니다!")
        print("감독님의 요구사항에 맞는 캡슐 방향제 제조가 가능합니다!")
    else:
        print("\n[FAIL] 시스템에 문제가 있습니다.")
    
    print("=" * 60)