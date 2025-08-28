#!/usr/bin/env python3
"""
딥러닝 모델 통합 시스템
훈련된 PyTorch 모델을 웹 인터페이스에 통합
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PerfumeNeuralNetwork(nn.Module):
    """향수 추천을 위한 딥러닝 모델 (복사본)"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        super(PerfumeNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class DeepLearningPerfumePredictor:
    """딥러닝 기반 향수 예측기"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "models/fragrance_dl_models/best_model.pth"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 및 전처리 도구들
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.is_loaded = False
        
        # 향료 노트 카테고리
        self.note_categories = {
            'citrus': ['citrus', 'bergamot', 'lemon', 'orange', 'grapefruit', 'lime', 'mandarin'],
            'floral': ['floral', 'rose', 'jasmine', 'lily', 'violet', 'tuberose', 'white floral', 'yellow floral'],
            'woody': ['woody', 'cedar', 'sandalwood', 'pine', 'cypress', 'guaiac wood'],
            'oriental': ['amber', 'vanilla', 'musk', 'oud', 'incense', 'benzoin'],
            'fresh': ['fresh', 'aquatic', 'green', 'herbal', 'mint', 'eucalyptus'],
            'spicy': ['warm spicy', 'fresh spicy', 'cinnamon', 'nutmeg', 'cardamom', 'black pepper'],
            'fruity': ['fruity', 'apple', 'berry', 'peach', 'plum', 'raspberry'],
            'gourmand': ['vanilla', 'chocolate', 'caramel', 'honey', 'coffee', 'almond', 'coconut'],
            'animalic': ['animalic', 'leather', 'musk', 'ambergris']
        }
        
        self.load_model_and_tools()
    
    def load_model_and_tools(self):
        """모델과 전처리 도구 로드"""
        try:
            # 메타데이터 로드
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # 전처리 도구 로드
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor_tools = pickle.load(f)
            
            # 모델 로드
            checkpoint = torch.load(self.model_path, map_location='cpu')
            model_config = checkpoint['model_config']
            
            self.model = PerfumeNeuralNetwork(
                input_dim=model_config['input_dim'],
                hidden_dims=model_config['hidden_dims'],
                output_dim=model_config['output_dim']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info("딥러닝 모델 및 전처리 도구 로드 완료")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            # 폴백: 기본값 설정
            self._setup_fallback()
    
    def _setup_fallback(self):
        """모델 로드 실패시 기본 설정"""
        logger.warning("딥러닝 모델 사용 불가 - 기본 모드로 전환")
        self.model = None
        self.preprocessor_tools = None
        self.metadata = {'feature_dim': 0, 'target_dim': 0}
    
    def predict_perfume_attributes(self, text_input: str, user_preferences: Optional[Dict] = None) -> Dict[str, Any]:
        """텍스트 입력으로부터 향수 속성 예측"""
        if self.model is None:
            return self._fallback_prediction(text_input)
        
        try:
            # 특성 추출
            features = self._extract_features_from_text(text_input, user_preferences)
            
            # 예측
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                predictions = self.model(features_tensor)
                predictions = predictions.squeeze(0).numpy()
            
            # 예측 결과 해석
            result = self._interpret_predictions(predictions, text_input)
            return result
            
        except Exception as e:
            logger.error(f"예측 실패: {e}")
            return self._fallback_prediction(text_input)
    
    def _extract_features_from_text(self, text: str, user_preferences: Optional[Dict] = None) -> np.ndarray:
        """텍스트에서 특성 추출"""
        if self.preprocessor_tools is None:
            # 기본 특성 반환
            return np.zeros(100)
        
        try:
            # TF-IDF 특성
            tfidf_vectorizer = self.preprocessor_tools['tfidf_vectorizer']
            text_features = tfidf_vectorizer.transform([text]).toarray().flatten()
            
            # 향료 노트 특성
            note_features = self._extract_note_features_from_text(text)
            
            # 사용자 선호도 특성 (있는 경우)
            user_features = self._extract_user_features(user_preferences)
            
            # 기타 기본 특성
            other_features = np.array([
                len(text),  # 텍스트 길이
                0,  # 성별 (기본값)
                4.0,  # 평점 (기본값)
                0,  # 평점 수 로그
                1   # 브랜드 (기본값)
            ])
            
            # 모든 특성 결합
            all_features = np.concatenate([
                other_features,
                note_features,
                text_features,
                user_features
            ])
            
            # 스케일링
            scaler = self.preprocessor_tools.get('scaler')
            if scaler and hasattr(scaler, 'transform'):
                # 올바른 차원으로 맞추기
                expected_dim = self.metadata.get('feature_dim', len(all_features))
                if len(all_features) < expected_dim:
                    # 부족한 특성은 0으로 채움
                    padding = np.zeros(expected_dim - len(all_features))
                    all_features = np.concatenate([all_features, padding])
                elif len(all_features) > expected_dim:
                    # 초과하는 특성은 잘라냄
                    all_features = all_features[:expected_dim]
                
                all_features = scaler.transform(all_features.reshape(1, -1)).flatten()
            
            return all_features
            
        except Exception as e:
            logger.error(f"특성 추출 실패: {e}")
            # 기본 특성 반환
            return np.zeros(self.metadata.get('feature_dim', 100))
    
    def _extract_note_features_from_text(self, text: str) -> np.ndarray:
        """텍스트에서 향료 노트 특성 추출"""
        text_lower = text.lower()
        note_features = []
        
        # 각 카테고리별 점수 계산
        for category, notes in self.note_categories.items():
            score = sum(1 for note in notes if note in text_lower)
            note_features.append(score)
        
        # 총 노트 수 추가
        total_notes = sum(note_features)
        note_features.append(total_notes)
        
        return np.array(note_features)
    
    def _extract_user_features(self, user_preferences: Optional[Dict]) -> np.ndarray:
        """사용자 선호도 특성 추출"""
        if not user_preferences:
            return np.array([0, 0, 0])  # 기본 사용자 특성
        
        # 강도 선호도, 성별 선호도, 계절 선호도 등
        intensity_pref = user_preferences.get('intensity', 5.0) / 10.0
        gender_pref = {'women': 0, 'men': 1, 'unisex': 2}.get(user_preferences.get('gender', 'unisex'), 2)
        season_pref = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}.get(user_preferences.get('season', 'spring'), 0)
        
        return np.array([intensity_pref, gender_pref, season_pref])
    
    def _interpret_predictions(self, predictions: np.ndarray, original_text: str) -> Dict[str, Any]:
        """예측 결과 해석"""
        result = {
            'predicted_rating': float(predictions[0]) if len(predictions) > 0 else 4.0,
            'gender_probabilities': {
                'women': float(predictions[1]) if len(predictions) > 1 else 0.33,
                'men': float(predictions[2]) if len(predictions) > 2 else 0.33,
                'unisex': float(predictions[3]) if len(predictions) > 3 else 0.34
            },
            'recommended_notes': self._generate_note_recommendations(original_text),
            'confidence': self._calculate_confidence(predictions),
            'ml_enhanced': True
        }
        
        # 성별 예측
        gender_probs = result['gender_probabilities']
        predicted_gender = max(gender_probs, key=gender_probs.get)
        result['predicted_gender'] = predicted_gender
        
        return result
    
    def _generate_note_recommendations(self, text: str) -> Dict[str, List[str]]:
        """텍스트 기반 노트 추천"""
        text_lower = text.lower()
        recommendations = {'top_notes': [], 'middle_notes': [], 'base_notes': []}
        
        # 키워드 기반 매칭
        keyword_mappings = {
            'fresh': {'top_notes': ['bergamot', 'lemon', 'mint'], 'middle_notes': ['green apple'], 'base_notes': []},
            'floral': {'top_notes': ['neroli'], 'middle_notes': ['rose', 'jasmine'], 'base_notes': []},
            'warm': {'top_notes': [], 'middle_notes': ['cinnamon', 'nutmeg'], 'base_notes': ['amber', 'sandalwood']},
            'sweet': {'top_notes': ['orange'], 'middle_notes': ['honey'], 'base_notes': ['vanilla', 'tonka bean']},
            'woody': {'top_notes': [], 'middle_notes': ['cedar'], 'base_notes': ['sandalwood', 'patchouli']},
            'spicy': {'top_notes': ['black pepper'], 'middle_notes': ['cardamom'], 'base_notes': ['clove']}
        }
        
        for keyword, notes in keyword_mappings.items():
            if keyword in text_lower:
                for note_type, note_list in notes.items():
                    recommendations[note_type].extend(note_list)
        
        # 중복 제거 및 기본값 설정
        for note_type in recommendations:
            recommendations[note_type] = list(set(recommendations[note_type]))
            if not recommendations[note_type]:
                # 기본 노트 추가
                if note_type == 'top_notes':
                    recommendations[note_type] = ['bergamot', 'lemon']
                elif note_type == 'middle_notes':
                    recommendations[note_type] = ['rose', 'jasmine']
                else:
                    recommendations[note_type] = ['musk', 'cedar']
        
        return recommendations
    
    def _calculate_confidence(self, predictions: np.ndarray) -> float:
        """예측 신뢰도 계산"""
        if len(predictions) == 0:
            return 0.5
        
        # 예측값들의 분산을 기반으로 신뢰도 계산
        variance = np.var(predictions)
        confidence = max(0.1, min(0.95, 1.0 / (1.0 + variance)))
        return confidence
    
    def _fallback_prediction(self, text: str) -> Dict[str, Any]:
        """딥러닝 모델 사용 불가시 폴백 예측"""
        logger.warning("딥러닝 예측 불가 - 기본 규칙 기반 예측 사용")
        
        # 간단한 키워드 기반 예측
        text_lower = text.lower()
        
        # 기본 평점 예측
        positive_words = ['좋은', '향기로운', '아름다운', 'beautiful', 'amazing', 'wonderful']
        rating = 4.0 + sum(0.2 for word in positive_words if word in text_lower)
        rating = min(5.0, rating)
        
        # 성별 예측
        female_words = ['여성', '꽃', 'floral', 'sweet', 'feminine', 'women']
        male_words = ['남성', 'woody', 'masculine', 'strong', 'men']
        
        female_score = sum(1 for word in female_words if word in text_lower)
        male_score = sum(1 for word in male_words if word in text_lower)
        
        if female_score > male_score:
            gender_probs = {'women': 0.6, 'men': 0.2, 'unisex': 0.2}
        elif male_score > female_score:
            gender_probs = {'women': 0.2, 'men': 0.6, 'unisex': 0.2}
        else:
            gender_probs = {'women': 0.33, 'men': 0.33, 'unisex': 0.34}
        
        return {
            'predicted_rating': rating,
            'gender_probabilities': gender_probs,
            'predicted_gender': max(gender_probs, key=gender_probs.get),
            'recommended_notes': self._generate_note_recommendations(text),
            'confidence': 0.6,
            'ml_enhanced': False,
            'fallback_mode': True
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_available': self.model is not None,
            'metadata': self.metadata,
            'model_path': str(self.model_path),
            'preprocessor_available': self.preprocessor_tools is not None
        }

class EnhancedPerfumeGenerator:
    """딥러닝이 통합된 향수 생성기"""
    
    def __init__(self, dl_predictor: DeepLearningPerfumePredictor):
        self.dl_predictor = dl_predictor
        
    def generate_enhanced_recipe(self, text_input: str, analysis: Dict, 
                               user_preferences: Optional[Dict] = None) -> Dict[str, Any]:
        """딥러닝 예측을 활용한 향상된 레시피 생성"""
        
        # 딥러닝 예측 수행
        dl_predictions = self.dl_predictor.predict_perfume_attributes(text_input, user_preferences)
        
        # 기존 분석과 딥러닝 예측 결합
        enhanced_recipe = {
            'name': self._generate_enhanced_name(analysis, dl_predictions),
            'description': self._generate_enhanced_description(text_input, analysis, dl_predictions),
            'top_notes': dl_predictions['recommended_notes']['top_notes'],
            'middle_notes': dl_predictions['recommended_notes']['middle_notes'], 
            'base_notes': dl_predictions['recommended_notes']['base_notes'],
            'intensity': self._calculate_intensity(analysis, dl_predictions),
            
            # 향상된 메타데이터
            'predicted_rating': dl_predictions['predicted_rating'],
            'predicted_gender': dl_predictions['predicted_gender'],
            'gender_probabilities': dl_predictions['gender_probabilities'],
            'ml_confidence': dl_predictions['confidence'],
            'ml_enhanced': dl_predictions['ml_enhanced'],
            
            # 조화도 계산
            'composition_harmony': self._calculate_harmony(dl_predictions['recommended_notes']),
            'confidence_scores': {
                'note_selection': dl_predictions['confidence'] * 0.8,
                'intensity': 0.7,
                'overall': dl_predictions['confidence']
            }
        }
        
        return enhanced_recipe
    
    def _generate_enhanced_name(self, analysis: Dict, dl_predictions: Dict) -> str:
        """딥러닝 예측을 반영한 향수 이름 생성"""
        # 감정과 성별 예측을 조합
        top_emotion = max(analysis.get('emotions', {'기쁨': 1}).items(), key=lambda x: x[1])[0]
        predicted_gender = dl_predictions['predicted_gender']
        
        name_templates = {
            'women': [f"우아한 {top_emotion}", f"{top_emotion}의 꽃", f"여성스러운 {top_emotion}"],
            'men': [f"강인한 {top_emotion}", f"{top_emotion}의 힘", f"남성적인 {top_emotion}"],
            'unisex': [f"{top_emotion}의 조화", f"자유로운 {top_emotion}", f"{top_emotion} 에센스"]
        }
        
        import random
        return random.choice(name_templates[predicted_gender])
    
    def _generate_enhanced_description(self, text_input: str, analysis: Dict, dl_predictions: Dict) -> str:
        """딥러닝 예측을 반영한 향수 설명 생성"""
        predicted_rating = dl_predictions['predicted_rating']
        confidence = dl_predictions['confidence']
        
        quality_desc = "뛰어난" if predicted_rating >= 4.5 else "훌륭한" if predicted_rating >= 4.0 else "좋은"
        
        description = f"{text_input[:100]}{'...' if len(text_input) > 100 else ''}의 감성을 담은 {quality_desc} 향수입니다. "
        
        if dl_predictions['ml_enhanced']:
            description += f"AI 분석을 통해 {confidence:.1%}의 신뢰도로 추천된 조합입니다."
        
        return description
    
    def _calculate_intensity(self, analysis: Dict, dl_predictions: Dict) -> float:
        """감정 분석과 딥러닝 예측을 결합한 강도 계산"""
        # 감정 강도
        emotion_intensity = sum(analysis.get('emotions', {}).values()) / len(analysis.get('emotions', {'default': 1}))
        
        # 예측된 평점을 강도로 변환
        rating_intensity = dl_predictions['predicted_rating'] / 5.0 * 10.0
        
        # 가중 평균
        final_intensity = (emotion_intensity * 0.6 + rating_intensity * 0.4)
        return max(1.0, min(10.0, final_intensity))
    
    def _calculate_harmony(self, notes: Dict[str, List[str]]) -> float:
        """노트 조합의 조화도 계산"""
        # 간단한 규칙 기반 조화도 계산
        total_notes = len(notes['top_notes']) + len(notes['middle_notes']) + len(notes['base_notes'])
        
        if total_notes == 0:
            return 0.5
        
        # 각 층의 균형
        balance_score = 1.0 - abs(len(notes['top_notes']) - len(notes['middle_notes']) - len(notes['base_notes'])) / total_notes
        
        # 노트 수가 적절한가
        count_score = 1.0 if 4 <= total_notes <= 8 else 0.7
        
        return (balance_score + count_score) / 2

# 새로운 훈련된 모델 통합 클래스
class TrainedFragrancePredictor:
    """200k 데이터셋으로 훈련된 딥러닝 모델 활용"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "models/fragrance_dl_models/best_model.pth"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.is_loaded = False
        
        logger.info("Trained fragrance predictor 초기화")
        self._load_trained_model()
        
    def _load_trained_model(self) -> bool:
        """훈련된 모델 로드"""
        model_file = Path(self.model_path)
        
        if not model_file.exists():
            logger.warning(f"Model file not found: {model_file}")
            return False
        
        try:
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            
            self.scaler = checkpoint.get('scaler')
            self.label_encoders = checkpoint.get('label_encoders', {})
            
            # FragranceNet 모델 로드
            from training.deep_learning_trainer import FragranceNet
            
            input_dim = 24
            output_dim = 5
            
            self.model = FragranceNet(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[1024, 512, 256, 128, 64]
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Trained model loaded successfully from {model_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            return False
    
    def predict_scene_fragrance(self, scene_description: str) -> Dict:
        """장면 설명으로부터 향기 특성 예측"""
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            # 장면 설명을 특성 벡터로 변환
            features = self._scene_to_features(scene_description)
            
            # 스케일링
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # 예측
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled).to(self.device)
                predictions = self.model(features_tensor)
                predictions = predictions.cpu().numpy().flatten()
            
            return {
                "success": True,
                "predictions": {
                    "intensity": max(0, min(100, float(predictions[0]))),
                    "longevity_hours": max(1, min(12, float(predictions[1]))),
                    "diffusion": max(1, min(10, float(predictions[2]))),
                    "threshold_ppb": max(0.1, float(predictions[3])),
                    "max_concentration": max(0.1, min(20, float(predictions[4])))
                },
                "model_enhanced": True
            }
            
        except Exception as e:
            logger.error(f"Scene prediction failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _scene_to_features(self, description: str) -> np.ndarray:
        """장면 설명을 특성 벡터로 변환"""
        features = []
        desc_lower = description.lower()
        
        # 기본 화학적 특성 (장면에서 추론)
        features.append(150.0)  # molecular_weight
        features.append(200.0 + (50 if "warm" in desc_lower else -20 if "cold" in desc_lower else 0))  # boiling_point
        features.append(0.9)    # density
        features.append(1.45)   # refractive_index
        features.append(60.0)   # flash_point
        features.append(85.0)   # solubility
        
        # 향기 계열 추론
        family_keywords = {
            "floral": ["flower", "rose", "jasmine", "romantic", "wedding", "garden"],
            "woody": ["forest", "wood", "tree", "cabin", "rustic", "nature"],
            "citrus": ["fresh", "morning", "bright", "sunny", "energetic"],
            "oriental": ["night", "mysterious", "exotic", "warm", "amber"],
            "fresh": ["ocean", "breeze", "clean", "water", "spring"],
            "fruity": ["sweet", "summer", "fruit", "young", "playful"]
        }
        
        detected_family = "floral"  # 기본값
        for family, keywords in family_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                detected_family = family
                break
        
        # 라벨 인코딩
        if detected_family in self.label_encoders.get('family', {}).classes_:
            features.append(float(self.label_encoders['family'].transform([detected_family])[0]))
        else:
            features.append(0.0)
        
        # 휘발성 추론
        volatility_keywords = {
            "top": ["fresh", "bright", "immediate", "burst"],
            "middle": ["heart", "main", "body", "core"], 
            "base": ["deep", "lasting", "foundation", "end"]
        }
        
        detected_volatility = "middle"
        for vol, keywords in volatility_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                detected_volatility = vol
                break
        
        if detected_volatility in self.label_encoders.get('volatility', {}).classes_:
            features.append(float(self.label_encoders['volatility'].transform([detected_volatility])[0]))
        else:
            features.append(1.0)
        
        # 추출 방법 (기본값)
        features.append(0.0)  # extraction method
        
        # 원산지 타입
        is_natural = any(word in desc_lower for word in ["natural", "garden", "forest", "flower", "plant"])
        features.append(1.0 if is_natural else 0.0)
        
        # 이진 특성들
        features.append(1.0 if is_natural else 0.0)  # is_natural
        features.append(0.0)  # ifra_restricted
        features.append(0.0)  # allergen
        
        # 경제적 특성 (장면 기반 추정)
        intensity_factor = 1.0
        if any(word in desc_lower for word in ["luxury", "expensive", "premium"]):
            intensity_factor = 2.0
        elif any(word in desc_lower for word in ["simple", "basic", "everyday"]):
            intensity_factor = 0.5
            
        features.append(500.0 * intensity_factor)  # price
        features.append(8.0)   # availability
        features.append(1000.0)  # production
        
        # 생산 특성
        features.append(75.0)  # yield
        features.append(95.0)  # purity
        features.append(12.0)  # processing_time
        
        # 응용 분야
        features.append(1.0)  # fine_fragrance
        features.append(1.0)  # personal_care
        features.append(0.0)  # home_care
        features.append(0.0)  # air_care
        
        # 복잡도 지표
        features.append(float(len(description.split())))
        
        return np.array(features, dtype=np.float32)

# 싱글톤 인스턴스
_trained_predictor = None

def get_trained_predictor() -> TrainedFragrancePredictor:
    """훈련된 예측기 싱글톤 인스턴스 반환"""
    global _trained_predictor
    if _trained_predictor is None:
        _trained_predictor = TrainedFragrancePredictor()
    return _trained_predictor