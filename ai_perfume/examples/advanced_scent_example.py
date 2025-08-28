import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.creative_scent_translator import CreativeScentTranslator
from core.scent_compatibility_checker import ScentCompatibilityChecker
from core.scent_evolution_simulator import ScentEvolutionSimulator
from core.fragrance_manufacturing_manager import FragranceManufacturingManager

def main():
    # 시스템 초기화
    translator = CreativeScentTranslator()
    compatibility_checker = ScentCompatibilityChecker()
    evolution_simulator = ScentEvolutionSimulator()
    manufacturer = FragranceManufacturingManager()
    
    # 예시 1: 영화 장면의 감성적 표현
    creative_description = """
    한밤중 비 내리는 도시의 거리.
    네온사인이 빗물에 반사되어 몽환적이고 신비로운 분위기를 자아내요.
    차가운 빗물 냄새와 아스팔트의 묵직한 향이 어우러지면서,
    마치 흑백 필름 누아르 영화의 한 장면 같은 분위기를 표현하고 싶어요.
    """
    
    print_example("예시 1: 영화 장면의 감성적 표현", creative_description)
    
    # 예시 2: 계절감과 시간성이 있는 표현
    creative_description = """
    이른 봄 아침, 안개가 자욱한 숲속 오솔길.
    아직 차가운 공기 속에서 막 피어나기 시작한 꽃들의 향기가 은은하게 퍼지고,
    젖은 이끼와 부드러운 흙냄새가 섞여 있어요.
    마치 동화 속 요정의 정원에 온 것 같은 신비로운 느낌이에요.
    """
    
    print_example("예시 2: 계절감과 시간성이 있는 표현", creative_description)

def print_example(title: str, description: str):
    # 시스템 초기화
    translator = CreativeScentTranslator()
    compatibility_checker = ScentCompatibilityChecker()
    evolution_simulator = ScentEvolutionSimulator()
    manufacturer = FragranceManufacturingManager()
    
    print("\n" + "="*80)
    print(f"\n{title}")
    print("\n[창의적 표현]")
    print(description.strip())
    
    # 1. 향 분석 및 변환
    analysis = translator.translate_creative_description(description)
    
    print("\n[분석된 창의적 요소]")
    if analysis['creative_elements']:
        for element in analysis['creative_elements']:
            print(f"- {element}")
    
    print("\n[향 구성]")
    print("\n1. 주요 향료:")
    for note in analysis['primary_notes']:
        print(f"- {note}")
    
    print("\n2. 보조 향료:")
    for note in analysis['supporting_notes']:
        print(f"- {note}")
    
    print("\n3. 베이스 향료:")
    for note in analysis['base_notes']:
        print(f"- {note}")
    
    # 2. 호환성 검사
    all_notes = analysis['primary_notes'] + analysis['supporting_notes'] + analysis['base_notes']
    compatibility_result = compatibility_checker.check_compatibility(all_notes)
    
    print("\n[호환성 분석]")
    print(f"호환성 점수: {compatibility_result['score']:.2f}/1.0")
    
    if compatibility_result['issues']:
        print("\n호환성 이슈:")
        for issue in compatibility_result['issues']:
            print(f"- {issue}")
    
    if compatibility_result['score'] < 0.7:
        print("\n개선 제안:")
        suggestions = compatibility_checker.suggest_improvements(all_notes)
        for suggestion in suggestions:
            print(f"- {suggestion['original_note']} → 대체 가능: {', '.join(suggestion['alternatives'])}")
    
    # 3. 향 발현 시뮬레이션
    recipe = {
        'top_notes': analysis['primary_notes'],
        'middle_notes': analysis['supporting_notes'],
        'base_notes': analysis['base_notes']
    }
    
    evolution_result = evolution_simulator.simulate_evolution(recipe)
    
    print("\n[향 발현 시뮬레이션]")
    print(f"\n지속성 점수: {evolution_result['longevity_score']:.1f}/10")
    print(f"확산력 점수: {evolution_result['sillage_score']:.1f}/10")
    print(f"발현 품질: {evolution_result['evolution_quality']['rating']}")
    
    print("\n시간대별 주요 향:")
    for t in evolution_result['timeline'][::4]:  # 4시간 간격으로 출력
        print(f"- {t['time']}: {', '.join(t['dominant_notes'])}")
    
    # 개선 제안이 있는 경우
    improvement_suggestions = evolution_simulator.get_improvement_suggestions(evolution_result)
    if improvement_suggestions:
        print("\n개선을 위한 제안:")
        for suggestion in improvement_suggestions:
            print(f"- {suggestion}")
    
    # 4. 제조 사양 생성
    spec = manufacturer.calculate_blend(recipe, 100)  # 100ml 기준
    
    print("\n[제조 사양]")
    print(f"\n총 용량: {spec['total_volume']}ml")
    print("\n원료별 용량:")
    for material in spec['materials']:
        print(f"- {material['name']}: {material['volume']:.1f}ml ({material['ratio']:.1f}%)")
    
    print(f"\n제조 방법: {spec['manufacturing_method']}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main() 