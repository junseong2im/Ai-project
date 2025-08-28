import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.creative_scent_translator import CreativeScentTranslator
from core.fragrance_manufacturing_manager import FragranceManufacturingManager

def main():
    # 시스템 초기화
    translator = CreativeScentTranslator()
    manufacturer = FragranceManufacturingManager()
    
    # 예시 1: 감정적이고 추상적인 표현
    creative_description = """
    마치 첫사랑의 설렘처럼 달콤하면서도 순수한 향.
    봄날의 햇살이 창문을 통해 스며드는 것처럼 따뜻하고 부드러운 느낌이 필요해요.
    살짝 시원한 바람이 불어오는 듯한 산뜻함도 있으면 좋겠어요.
    """
    
    print_example("예시 1: 감정적이고 추상적인 표현", creative_description)
    
    # 예시 2: 영화 장면 묘사
    creative_description = """
    한밤중 비가 내리는 도시의 거리.
    네온사인이 빗물에 반사되어 몽환적이고 신비로운 분위기를 자아내요.
    차가운 빗물 냄새와 아스팔트의 묵직한 향이 어우러지면서,
    마치 흑백 필름 누아르 영화의 한 장면 같은 분위기를 표현하고 싶어요.
    """
    
    print_example("예시 2: 영화 장면 묘사", creative_description)
    
    # 예시 3: 자연과 계절감 표현
    creative_description = """
    한여름 새벽, 안개가 자욱한 숲속 오솔길.
    이슬을 머금은 풀잎에서는 신선한 허브향이 나고,
    멀리서 들려오는 바다 내음이 산들바람에 실려와요.
    마치 요정이 사는 숲속에 들어온 것 같은 신비로운 느낌이에요.
    """
    
    print_example("예시 3: 자연과 계절감 표현", creative_description)

def print_example(title: str, description: str):
    translator = CreativeScentTranslator()
    manufacturer = FragranceManufacturingManager()
    
    print("\n" + "="*80)
    print(f"\n{title}")
    print("\n[창의적 표현]")
    print(description.strip())
    
    # 향 분석 및 변환
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
    
    print("\n[향의 특성]")
    print("- 특성:", ", ".join(analysis['characteristics']))
    print("- 질감:", ", ".join(analysis['texture']))
    print(f"- 강도: {analysis['intensity']:.1f}/10")
    
    # 제조 사양 생성
    materials = []
    # Convert note names to materials with concentrations
    for note in analysis['primary_notes']:
        materials.append({'name': note, 'concentration': 15.0})  # 15% each
    for note in analysis['supporting_notes']:
        materials.append({'name': note, 'concentration': 10.0})  # 10% each
    for note in analysis['base_notes']:
        materials.append({'name': note, 'concentration': 5.0})   # 5% each
    
    spec = manufacturer.calculate_blend(materials, 100)  # 100ml 기준
    
    print("\n[제조 사양]")
    print(f"\n총 용량: {spec['total_volume']}ml")
    print("\n원료별 용량:")
    for material in spec['materials']:
        print(f"- {material['name']}: {material['volume']:.1f}ml ({material['ratio']:.1f}%)")
    
    print(f"\n제조 방법: {spec['manufacturing_method']}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main() 