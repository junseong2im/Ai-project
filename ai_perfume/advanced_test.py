#!/usr/bin/env python3
"""
고급 딥러닝 통합 시스템 테스트
"""

import sys
from pathlib import Path
import time

# 현재 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent))

from models.text_analyzer import TextAnalyzer
from core.deep_learning_integration import DeepLearningPerfumePredictor, EnhancedPerfumeGenerator

def advanced_perfume_test():
    """고급 향수 테스트"""
    
    print("=" * 60)
    print("딥러닝 통합 향수 AI 고급 테스트")
    print("=" * 60)
    
    # 시스템 초기화
    text_analyzer = TextAnalyzer()
    dl_predictor = DeepLearningPerfumePredictor(
        "models/perfume_dl_model.pth",
        "data/processed/preprocessor_tools.pkl",
        "data/processed/metadata.json"
    )
    enhanced_generator = EnhancedPerfumeGenerator(dl_predictor)
    
    # 다양한 테스트 시나리오
    test_scenarios = [
        {
            'text': "여름 저녁 해변에서 바라본 석양과 함께 느끼는 로맨틱한 감정",
            'preferences': {'intensity': 7, 'gender': 'women', 'season': 'summer'},
            'title': "로맨틱 해변 시나리오"
        },
        {
            'text': "겨울 산속 오두막에서 느끼는 따뜻하고 아늑한 분위기",
            'preferences': {'intensity': 8, 'gender': 'men', 'season': 'winter'},  
            'title': "겨울 오두막 시나리오"
        },
        {
            'text': "봄날 커피숍에서 책을 읽으며 느끼는 평온하고 집중된 시간",
            'preferences': {'intensity': 5, 'gender': 'unisex', 'season': 'spring'},
            'title': "봄날 독서 시나리오"
        },
        {
            'text': "가을 공원에서 단풍을 보며 느끼는 노스탤지어와 그리움",
            'preferences': {'intensity': 6, 'gender': 'women', 'season': 'autumn'},
            'title': "가을 노스탤지어 시나리오"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n[테스트 {i}] {scenario['title']}")
        print(f"입력: {scenario['text']}")
        print(f"선호도: {scenario['preferences']}")
        print("-" * 50)
        
        start_time = time.time()
        
        # 분석 및 생성
        analysis = text_analyzer.analyze(scenario['text'])
        recipe = enhanced_generator.generate_enhanced_recipe(
            scenario['text'], analysis, scenario['preferences']
        )
        
        processing_time = time.time() - start_time
        
        # 결과 출력
        print(f"향수명: {recipe['name']}")
        print(f"설명: {recipe['description'][:100]}...")
        print(f"노트 구성:")
        print(f"  - 탑: {', '.join(recipe['top_notes'])}")
        print(f"  - 미들: {', '.join(recipe['middle_notes'])}")
        print(f"  - 베이스: {', '.join(recipe['base_notes'])}")
        print(f"강도: {recipe['intensity']:.1f}")
        print(f"예측 평점: {recipe['predicted_rating']:.2f}")
        print(f"예측 성별: {recipe['predicted_gender']}")
        print(f"처리시간: {processing_time:.2f}초")
        
        results.append({
            'scenario': scenario['title'],
            'recipe': recipe,
            'processing_time': processing_time
        })
    
    # 전체 통계
    print(f"\n{'='*60}")
    print("전체 테스트 결과 통계")
    print("="*60)
    
    avg_time = sum(r['processing_time'] for r in results) / len(results)
    avg_rating = sum(r['recipe']['predicted_rating'] for r in results) / len(results)
    avg_intensity = sum(r['recipe']['intensity'] for r in results) / len(results)
    
    print(f"평균 처리 시간: {avg_time:.2f}초")
    print(f"평균 예측 평점: {avg_rating:.2f}/5.0")  
    print(f"평균 강도: {avg_intensity:.1f}/10")
    
    # 성별 분포
    gender_dist = {}
    for r in results:
        gender = r['recipe']['predicted_gender']
        gender_dist[gender] = gender_dist.get(gender, 0) + 1
    
    print(f"성별 분포: {gender_dist}")
    
    # 가장 많이 사용된 노트
    all_notes = []
    for r in results:
        all_notes.extend(r['recipe']['top_notes'])
        all_notes.extend(r['recipe']['middle_notes']) 
        all_notes.extend(r['recipe']['base_notes'])
    
    from collections import Counter
    top_notes = Counter(all_notes).most_common(5)
    print(f"인기 노트 TOP 5: {top_notes}")
    
    print(f"\n총 {len(results)}개 시나리오 테스트 완료!")
    print("딥러닝 통합 향수 AI 시스템이 성공적으로 작동합니다!")

if __name__ == "__main__":
    advanced_perfume_test()