from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.scene_scent_matcher import SceneScentMatcher


def main() -> None:
    """메인 실행 함수"""
    # 시스템 초기화
    matcher = SceneScentMatcher()
    
    # 예시 1: 장면 설명만 있는 경우
    scene_description = """
    비 오는 새벽, 오래된 도서관의 나무 책장에서 묵직한 가죽 장정본을 꺼내는 장면.
    습기를 머금은 공기와 오래된 책의 냄새가 섞여 있다.
    """
    
    result = matcher.process_scene_description(scene_description)
    print_result("예시 1: 장면 설명만 있는 경우", scene_description, result)
    
    # 예시 2: 레퍼런스 향이 있는 경우
    scene_description = """
    한여름 저녁, 해변가 절벽 위 등대. 
    짙은 주황빛 노을이 지는 가운데, 시원한 바닷바람이 불어온다.
    소금기를 머금은 공기가 피부에 닿는다.
    """
    
    reference_scents = [
        "시원하고 깨끗한 해변의 향. 소금기가 느껴지는 시트러스 향이 강하다.",
        "신선한 오존 향에 우디한 베이스가 깔려있다."
    ]
    
    result = matcher.process_scene_description(scene_description, reference_scents)
    print_result("예시 2: 레퍼런스 향이 있는 경우", scene_description, result, reference_scents)

def print_result(
    title: str, 
    scene: str, 
    result: Dict[str, Any], 
    references: Optional[List[str]] = None
) -> None:
    """결과 출력 함수"""
    print("\n" + "="*80)
    print(f"\n{title}")
    print("\n장면 설명:")
    print(scene.strip())
    
    if references:
        print("\n레퍼런스 향:")
        for i, ref in enumerate(references, 1):
            print(f"{i}. {ref}")
    
    print("\n분석 결과:")
    analysis = result['analysis']
    print("\n1. 향 프로파일:")
    for category, score in analysis['scent_profile']:
        print(f"- {category}: {score:.2f}")
    
    print(f"\n2. 강도: {analysis['intensity']['description']} ({analysis['intensity']['level']:.1f}/10)")
    
    print("\n3. 질감:", ", ".join(analysis['texture']) if analysis['texture'] else "없음")
    
    print("\n4. 환경 요소:")
    for key, value in analysis['environment'].items():
        if value:
            print(f"- {key}: {value}")
    
    print("\n제조 사양:")
    spec = result['manufacturing']['materials']
    print(f"\n총 용량: {result['manufacturing']['total_volume']}ml")
    print("\n원료 구성:")
    for material in spec:
        print(f"- {material['name']}: {material['volume']}ml ({material['ratio']}%)")
    
    print(f"\n제조 방법: {result['manufacturing']['manufacturing_method']}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main() 