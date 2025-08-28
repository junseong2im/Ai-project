#!/usr/bin/env python3
"""
딥러닝 통합 시스템 간단 테스트
"""

import sys
from pathlib import Path

# 현재 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent))

from models.text_analyzer import TextAnalyzer
from core.deep_learning_integration import DeepLearningPerfumePredictor, EnhancedPerfumeGenerator

def test_deep_learning_system():
    """딥러닝 시스템 테스트"""
    print("=== 딥러닝 통합 향수 AI 시스템 테스트 ===")
    
    try:
        # 딥러닝 예측기 초기화
        model_path = "models/perfume_dl_model.pth"
        preprocessor_path = "data/processed/preprocessor_tools.pkl"  
        metadata_path = "data/processed/metadata.json"
        
        print("1. 딥러닝 모델 로딩 중...")
        dl_predictor = DeepLearningPerfumePredictor(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            metadata_path=metadata_path
        )
        
        print("2. 향상된 생성기 초기화 중...")
        enhanced_generator = EnhancedPerfumeGenerator(dl_predictor)
        
        print("3. 텍스트 분석기 초기화 중...")
        text_analyzer = TextAnalyzer()
        
        # 테스트 입력
        test_text = "봄날 아침 정원에서 피어나는 장미와 함께 느끼는 평온하고 행복한 순간"
        user_preferences = {'intensity': 6, 'gender': 'women', 'season': 'spring'}
        
        print(f"4. 테스트 텍스트: {test_text}")
        print(f"5. 사용자 선호도: {user_preferences}")
        
        # 텍스트 분석
        print("6. 텍스트 분석 중...")
        analysis = text_analyzer.analyze(test_text)
        
        # 딥러닝 예측
        print("7. 딥러닝 예측 수행 중...")
        dl_predictions = dl_predictor.predict_perfume_attributes(test_text, user_preferences)
        
        # 향상된 레시피 생성
        print("8. 향상된 레시피 생성 중...")
        enhanced_recipe = enhanced_generator.generate_enhanced_recipe(
            test_text, analysis, user_preferences
        )
        
        # 결과 출력
        print("\n=== 결과 ===")
        print(f"향수명: {enhanced_recipe['name']}")
        print(f"설명: {enhanced_recipe['description']}")
        print(f"탑 노트: {', '.join(enhanced_recipe['top_notes'])}")
        print(f"미들 노트: {', '.join(enhanced_recipe['middle_notes'])}")
        print(f"베이스 노트: {', '.join(enhanced_recipe['base_notes'])}")
        print(f"강도: {enhanced_recipe['intensity']:.1f}/10")
        
        # 딥러닝 예측 정보
        print(f"\n=== 딥러닝 예측 정보 ===")
        print(f"예측 평점: {enhanced_recipe['predicted_rating']:.2f}/5.0")
        print(f"예측 성별: {enhanced_recipe['predicted_gender']}")
        print(f"ML 신뢰도: {enhanced_recipe['ml_confidence']:.1%}")
        print(f"조화도: {enhanced_recipe['composition_harmony']:.2f}")
        
        # 모델 정보
        print(f"\n=== 모델 정보 ===")
        model_info = dl_predictor.get_model_info()
        print(f"모델 사용 가능: {model_info['model_available']}")
        print(f"입력 차원: {model_info['metadata'].get('feature_dim', 'N/A')}")
        print(f"출력 차원: {model_info['metadata'].get('target_dim', 'N/A')}")
        
        print("\n테스트 완료 - 딥러닝 시스템이 성공적으로 작동합니다!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deep_learning_system()