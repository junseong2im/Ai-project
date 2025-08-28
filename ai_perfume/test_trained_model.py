#!/usr/bin/env python3
"""
훈련된 영화 향료 AI 모델 테스트
다양한 영화 장면으로 정확도 검증
"""

import sys
import torch
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Any
import json

# 상위 디렉토리 추가
sys.path.append(str(Path(__file__).parent))

from training.movie_scent_trainer import MovieScentNeuralNetwork

class TrainedMovieScentPredictor:
    """훈련된 영화 향료 예측기"""
    
    def __init__(self, model_dir: str = "models/movie_scent_models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessors = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model_and_preprocessors()
    
    def _load_model_and_preprocessors(self):
        """모델과 전처리기 로드"""
        # 모델 로드
        model_path = self.model_dir / "movie_scent_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        self.model = MovieScentNeuralNetwork(
            input_dim=model_config['input_dim'],
            material_count=model_config['material_count'],
            genre_count=model_config['genre_count'],
            emotion_count=model_config['emotion_count']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 전처리기 로드
        preprocessor_path = self.model_dir / "preprocessors.pkl"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessors not found at {preprocessor_path}")
        
        self.preprocessors = joblib.load(preprocessor_path)
        
        print(f"Model loaded successfully!")
        print(f"   - Material vocabulary size: {len(self.preprocessors['material_vocab'])}")
        print(f"   - Emotion vocabulary size: {len(self.preprocessors['emotion_vocab'])}")
    
    def predict_scene_fragrance(self, scene_description: str, genre: str = "drama") -> Dict[str, Any]:
        """영화 장면으로부터 향료 조합 예측"""
        
        # 1. 특성 추출
        features = self._extract_scene_features(scene_description, genre)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # 2. 모델 예측
        with torch.no_grad():
            outputs = self.model(features_tensor)
        
        # 3. 결과 해석
        result = self._interpret_predictions(outputs, scene_description)
        
        return result
    
    def _extract_scene_features(self, scene_description: str, genre: str) -> List[float]:
        """장면으로부터 특성 벡터 추출 (훈련 시와 동일한 방식)"""
        features = []
        
        # 1. 장르 원-핫 인코딩
        genres = ['action', 'romantic', 'horror', 'drama', 'thriller', 'comedy', 'sci_fi']
        genre_vector = [1.0 if genre == g else 0.0 for g in genres]
        features.extend(genre_vector)
        
        # 2. 텍스트 특성
        text_features = self._extract_text_features(scene_description)
        features.extend(text_features)
        
        # 3. 더미 메타데이터 특성
        features.append(0.5)  # 정규화된 recipe_id
        
        # 4. 특성 정규화
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.preprocessors['feature_scaler'].transform(features_array)
        
        return features_scaled[0].tolist()
    
    def _extract_text_features(self, text: str) -> List[float]:
        """텍스트에서 특성 추출"""
        text_lower = text.lower()
        
        emotion_keywords = {
            'action': ['explosion', 'chase', 'fight', 'battle', 'war', 'gun', 'car'],
            'romantic': ['love', 'kiss', 'couple', 'romantic', 'heart', 'tender'],
            'scary': ['dark', 'scary', 'horror', 'fear', 'blood', 'nightmare'],
            'peaceful': ['calm', 'quiet', 'peaceful', 'serene', 'meditation'],
            'nature': ['forest', 'ocean', 'mountain', 'garden', 'flower', 'tree'],
            'urban': ['city', 'street', 'building', 'office', 'hotel', 'club'],
            'temporal': ['morning', 'night', 'sunset', 'dawn', 'evening', 'midnight']
        }
        
        features = []
        for category, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
            features.append(score)
        
        # 텍스트 길이 특성
        features.append(len(text.split()) / 20.0)
        
        return features
    
    def _interpret_predictions(self, outputs: Dict[str, torch.Tensor], scene_description: str) -> Dict[str, Any]:
        """모델 출력을 해석"""
        # CPU로 이동 및 numpy 변환
        materials = outputs['materials'].cpu().numpy()[0]
        volatility = outputs['volatility'].cpu().numpy()[0]
        duration = outputs['duration'].cpu().numpy()[0][0]
        emotions = outputs['emotions'].cpu().numpy()[0]
        
        # 재료 분석 (상위 5개)
        material_vocab = self.preprocessors['material_vocab']
        material_names = list(material_vocab.keys())
        
        # 농도가 높은 재료들 선택
        material_indices = np.argsort(materials)[::-1][:8]  # 상위 8개
        selected_materials = []
        
        for idx in material_indices:
            concentration = materials[idx] * 100  # 백분율로 변환
            if concentration > 0.1:  # 0.1% 이상만 선택
                selected_materials.append({
                    'name': material_names[idx],
                    'concentration_percent': round(concentration, 2),
                    'predicted_strength': 'high' if concentration > 5 else 'medium' if concentration > 1 else 'low'
                })
        
        # 휘발성 레벨
        volatility_levels = ['low_volatility', 'medium_volatility', 'high_volatility']
        predicted_volatility = volatility_levels[np.argmax(volatility)]
        
        # 감정 분석 (확률 > 0.5인 감정들)
        emotion_vocab = self.preprocessors['emotion_vocab']
        emotion_names = list(emotion_vocab.keys())
        predicted_emotions = []
        
        for i, prob in enumerate(emotions):
            if prob > 0.5:
                predicted_emotions.append({
                    'emotion': emotion_names[i],
                    'confidence': round(prob, 3)
                })
        
        # 노트별 분류
        top_notes = []
        middle_notes = []
        base_notes = []
        
        for material in selected_materials:
            name = material['name']
            # 간단한 휴리스틱으로 노트 분류
            if any(x in name for x in ['lemon', 'bergamot', 'orange', 'lime', 'peppermint', 'eucalyptus']):
                top_notes.append(material)
            elif any(x in name for x in ['sandalwood', 'cedar', 'vanilla', 'musk', 'amber', 'patchouli']):
                base_notes.append(material)
            else:
                middle_notes.append(material)
        
        # 지속시간 해석
        duration_seconds = max(10.0, min(600.0, float(duration)))  # 10초~10분 범위
        if duration_seconds < 30:
            duration_str = f"{int(duration_seconds)}초 (즉석 효과)"
        elif duration_seconds < 120:
            duration_str = f"{int(duration_seconds)}초 (단기 지속)"
        else:
            duration_str = f"{int(duration_seconds/60)}분 (중기 지속)"
        
        result = {
            'scene_description': scene_description,
            'predicted_volatility': predicted_volatility,
            'predicted_duration': duration_str,
            'predicted_emotions': predicted_emotions,
            'fragrance_notes': {
                'top_notes': top_notes,
                'middle_notes': middle_notes,
                'base_notes': base_notes
            },
            'total_materials': len(selected_materials),
            'confidence_score': self._calculate_confidence(outputs),
            'recommendation_summary': self._generate_summary(selected_materials, predicted_volatility, predicted_emotions)
        }
        
        return result
    
    def _calculate_confidence(self, outputs: Dict[str, torch.Tensor]) -> float:
        """예측 신뢰도 계산"""
        # 각 출력의 확실성을 기반으로 신뢰도 계산
        materials = outputs['materials'].cpu().numpy()[0]
        volatility = outputs['volatility'].cpu().numpy()[0]
        emotions = outputs['emotions'].cpu().numpy()[0]
        
        # 재료 농도의 분산 (높을수록 특정 재료에 집중됨)
        material_confidence = np.std(materials)
        
        # 휘발성 예측의 확실성
        volatility_confidence = np.max(torch.softmax(outputs['volatility'], dim=1).cpu().numpy()[0])
        
        # 감정 예측의 확실성
        emotion_confidence = np.mean(np.abs(emotions - 0.5)) * 2  # 0.5에서 멀수록 확실
        
        # 종합 신뢰도
        confidence = (material_confidence * 0.4 + volatility_confidence * 0.3 + emotion_confidence * 0.3)
        return min(0.95, max(0.1, confidence))
    
    def _generate_summary(self, materials: List[Dict], volatility: str, emotions: List[Dict]) -> str:
        """추천 요약 생성"""
        if not materials:
            return "적합한 향료 조합을 찾지 못했습니다."
        
        main_materials = [m['name'].replace('_', ' ').title() for m in materials[:3]]
        material_str = ', '.join(main_materials)
        
        volatility_str = {
            'high_volatility': '빠른 확산',
            'medium_volatility': '보통 확산', 
            'low_volatility': '은은한 확산'
        }.get(volatility, '보통 확산')
        
        emotion_str = ', '.join([e['emotion'].title() for e in emotions[:2]]) if emotions else 'Neutral'
        
        summary = f"주요 원료: {material_str} | 확산성: {volatility_str} | 감정: {emotion_str}"
        return summary

def test_various_movie_scenes():
    """다양한 영화 장면으로 모델 테스트"""
    
    print("Trained Movie Scent AI - Model Testing")
    print("=" * 60)
    
    try:
        predictor = TrainedMovieScentPredictor()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the model is trained first!")
        return
    
    # 테스트 장면들
    test_scenes = [
        {
            "description": "로맨틱한 해변가 석양에서 커플이 키스하는 장면",
            "genre": "romantic"
        },
        {
            "description": "액션 영화의 폭발 장면에서 주인공이 달려나오는 모습",
            "genre": "action"
        },
        {
            "description": "공포 영화의 어두운 지하실에서 괴물과 마주치는 순간",
            "genre": "horror"
        },
        {
            "description": "조용한 도서관에서 혼자 책을 읽는 평화로운 오후",
            "genre": "drama"
        },
        {
            "description": "미래 도시의 네온사인 거리를 걸어가는 사이버펑크 장면",
            "genre": "sci_fi"
        },
        {
            "description": "스릴러 영화의 긴장감 넘치는 추격전 장면",
            "genre": "thriller"
        }
    ]
    
    for i, scene in enumerate(test_scenes, 1):
        print(f"\n[테스트 {i}] {scene['genre'].upper()}")
        print(f"장면: {scene['description']}")
        print("-" * 50)
        
        try:
            result = predictor.predict_scene_fragrance(
                scene['description'], 
                scene['genre']
            )
            
            print(f"예측 결과:")
            print(f"   휘발성: {result['predicted_volatility']}")
            print(f"   지속시간: {result['predicted_duration']}")
            print(f"   신뢰도: {result['confidence_score']:.2f}")
            
            if result['predicted_emotions']:
                emotions_str = ', '.join([f"{e['emotion']}({e['confidence']:.2f})" for e in result['predicted_emotions']])
                print(f"   감정: {emotions_str}")
            
            print(f"향료 구성 ({result['total_materials']}종):")
            
            for note_type, notes in result['fragrance_notes'].items():
                if notes:
                    type_name = note_type.replace('_', ' ').title()
                    print(f"   {type_name}: ", end="")
                    note_strs = [f"{note['name']} {note['concentration_percent']}%" for note in notes]
                    print(', '.join(note_strs))
            
            print(f"요약: {result['recommendation_summary']}")
            
        except Exception as e:
            print(f"예측 실패: {e}")
    
    print("\n" + "=" * 60)
    print("모든 테스트 완료!")

if __name__ == "__main__":
    test_various_movie_scenes()