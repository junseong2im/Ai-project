#!/usr/bin/env python3
"""
실시간 영화 장면 향수 추천 시스템
"""

import torch
import numpy as np
import json
import pickle
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import time
from collections import defaultdict

# 실제 딥러닝 모델 임포트
try:
    from .movie_scent_ai import MovieScentAI
except ImportError:
    try:
        from movie_scent_ai import MovieScentAI
    except ImportError:
        logger.warning("MovieScentAI를 임포트할 수 없습니다. 규칙 기반 시스템만 사용됩니다.")
        MovieScentAI = None

logger = logging.getLogger(__name__)

class RealTimeMovieScentRecommender:
    """실시간 영화 장면 향수 추천기"""
    
    def __init__(self, model_path: str = "models/ultimate_movie_scent_model.pth",
                 preprocessor_path: str = "models/movie_scent_preprocessor.pkl"):
        self.model = None
        self.preprocessor = None
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        
        # 실제 딥러닝 AI 모델
        self.movie_ai = None
        
        # 실시간 캐시
        self.scene_cache = {}
        self.recommendation_cache = {}
        
        # 성능 통계
        self.stats = {
            'total_recommendations': 0,
            'cache_hits': 0,
            'average_response_time': 0.0
        }
        
        # 향수 브랜드 데이터베이스
        self.perfume_database = {
            'romantic': {
                'Chanel': ['No.5', 'Coco Mademoiselle', 'Chance'],
                'Dior': ['Miss Dior', 'J\'adore', 'Blooming Bouquet'],
                'Tom Ford': ['Black Orchid', 'Orchid Soleil', 'Rose Prick']
            },
            'intense': {
                'Tom Ford': ['Tobacco Vanille', 'Oud Wood', 'Black Orchid'],
                'Creed': ['Aventus', 'Silver Mountain Water', 'Green Irish Tweed'],
                'Maison Margiela': ['By the Fireplace', 'Jazz Club', 'Coffee Break']
            },
            'fresh': {
                'Acqua di Parma': ['Colonia', 'Blu Mediterraneo', 'Arancia di Capri'],
                'Hermès': ['Un Jardin Sur Le Toit', 'Eau des Merveilles', 'Terre d\'Hermès'],
                'L\'Occitane': ['Verbena', 'Lavender', 'Immortelle']
            },
            'sophisticated': {
                'Chanel': ['Allure Homme', 'Bleu de Chanel', 'Platinum Égoïste'],
                'Giorgio Armani': ['Acqua di Gio', 'Code', 'Privé Collection'],
                'Yves Saint Laurent': ['La Nuit de l\'Homme', 'L\'Homme', 'Opium']
            }
        }
        
        logger.info("실시간 영화 향수 추천기 초기화 완료")
    
    def load_model_and_preprocessor(self):
        """모델 및 전처리기 로드"""
        try:
            # 1. 기본 모델 로드 시도
            if self.model_path.exists():
                # 기존 방식으로 모델 로드
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                # 모델 재구성
                from core.movie_scent_ai import AdvancedMovieNeuralNetwork
                model_config = checkpoint['model_config']
                
                self.model = AdvancedMovieNeuralNetwork(
                    input_dim=model_config['input_dim'],
                    hidden_dims=model_config['hidden_dims'],
                    output_dim=model_config['output_dim'],
                    num_heads=16,
                    use_attention=True,
                    use_residual=True
                )
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                logger.info("딥러닝 모델 로드 완료")
            
            # 2. MovieScentAI 시스템 로드 시도 (우선순위)
            if MovieScentAI:
                try:
                    self.movie_ai = MovieScentAI()
                    
                    # 모델 파일이 존재하면 로드
                    model_files = list(Path("models").glob("*movie*model*.pth"))
                    if model_files:
                        # 가장 최근 모델 파일 선택
                        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                        logger.info(f"MovieScentAI 모델 로드 시도: {latest_model}")
                        
                        # 모델 로드 (MovieScentAI 내부에서 처리)
                        checkpoint = torch.load(latest_model, map_location='cpu')
                        if 'model_state_dict' in checkpoint:
                            # 모델 구조 재생성
                            from core.movie_scent_ai import AdvancedMovieNeuralNetwork
                            model_config = checkpoint.get('model_config', {
                                'input_dim': 155,
                                'hidden_dims': [1024, 512, 256, 128, 64],
                                'output_dim': 18  # 3 + 15 categories
                            })
                            
                            self.movie_ai.model = AdvancedMovieNeuralNetwork(
                                input_dim=model_config['input_dim'],
                                hidden_dims=model_config['hidden_dims'],
                                output_dim=model_config['output_dim'],
                                num_heads=16,
                                use_attention=True,
                                use_residual=True
                            )
                            
                            self.movie_ai.model.load_state_dict(checkpoint['model_state_dict'])
                            self.movie_ai.model.eval()
                            logger.info("✅ MovieScentAI 고급 모델 로드 성공!")
                        
                    else:
                        # 모델 파일이 없으면 기본 초기화
                        logger.info("모델 파일이 없어 MovieScentAI 기본 모드로 초기화")
                        
                except Exception as e:
                    logger.error(f"MovieScentAI 로드 실패: {e}")
                    self.movie_ai = None
            
            # 3. 전처리기 로드
            if self.preprocessor_path.exists():
                with open(self.preprocessor_path, 'rb') as f:
                    preprocessor_data = pickle.load(f)
                
                self.preprocessor = preprocessor_data
                logger.info("전처리기 로드 완료")
            
            # 성공 여부 반환
            has_basic_model = self.model is not None
            has_ai_model = self.movie_ai is not None and hasattr(self.movie_ai, 'model') and self.movie_ai.model is not None
            
            if has_ai_model:
                logger.info("🚀 MovieScentAI 고급 모델 준비 완료!")
                return True
            elif has_basic_model:
                logger.info("✅ 기본 모델 준비 완료")
                return True
            else:
                logger.warning("⚠️ 모델 로드 실패, 규칙 기반 시스템 사용")
                return False
            
        except Exception as e:
            logger.error(f"모델/전처리기 로드 실패: {e}")
            return False
    
    def recommend_for_scene(self, scene_description: str, 
                           scene_type: str = "drama",
                           mood: str = "neutral",
                           intensity_preference: int = 5) -> Dict[str, Any]:
        """영화 장면에 대한 실시간 향수 추천"""
        
        start_time = time.time()
        self.stats['total_recommendations'] += 1
        
        # 캐시 체크
        cache_key = f"{scene_description}_{scene_type}_{mood}_{intensity_preference}"
        if cache_key in self.recommendation_cache:
            self.stats['cache_hits'] += 1
            logger.info("캐시에서 추천 결과 반환")
            return self.recommendation_cache[cache_key]
        
        try:
            # 장면 분석
            scene_data = self._analyze_scene(scene_description, scene_type, mood)
            
            # 향수 예측 (모델이 있는 경우)
            if self.model and self.preprocessor:
                prediction = self._predict_with_model(scene_data)
            else:
                prediction = self._fallback_prediction(scene_data, intensity_preference)
            
            # 구체적인 향수 제품 추천
            product_recommendations = self._recommend_specific_perfumes(prediction, scene_type, mood)
            
            # 최종 결과 구성
            result = {
                'scene_analysis': scene_data,
                'scent_profile': prediction,
                'product_recommendations': product_recommendations,
                'meta': {
                    'response_time': time.time() - start_time,
                    'confidence': prediction.get('confidence', 0.8),
                    'model_used': self.model is not None,
                    'cache_used': False
                }
            }
            
            # 캐시에 저장 (최대 100개)
            if len(self.recommendation_cache) < 100:
                self.recommendation_cache[cache_key] = result
            
            # 성능 통계 업데이트
            self._update_stats(time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"추천 생성 실패: {e}")
            return self._emergency_fallback(scene_description, scene_type)
    
    def _analyze_scene(self, description: str, scene_type: str, mood: str) -> Dict[str, Any]:
        """장면 분석"""
        
        # 키워드 기반 분석
        description_lower = description.lower()
        
        # 위치 추정
        location_keywords = {
            'beach': ['beach', 'ocean', 'sea', 'sand', 'waves'],
            'forest': ['forest', 'trees', 'woods', 'pine', 'nature'],
            'city': ['city', 'street', 'building', 'urban', 'downtown'],
            'home': ['home', 'house', 'room', 'kitchen', 'bedroom'],
            'restaurant': ['restaurant', 'cafe', 'dinner', 'food', 'wine']
        }
        
        detected_location = 'unknown'
        for location, keywords in location_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_location = location
                break
        
        # 시간대 추정
        time_keywords = {
            'morning': ['morning', 'dawn', 'sunrise', 'breakfast'],
            'afternoon': ['afternoon', 'lunch', 'noon', 'day'],
            'evening': ['evening', 'dinner', 'sunset', 'dusk'],
            'night': ['night', 'midnight', 'dark', 'sleep']
        }
        
        detected_time = 'unknown'
        for time_period, keywords in time_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_time = time_period
                break
        
        # 감정 분석
        emotion_keywords = {
            'love': ['love', 'romantic', 'kiss', 'heart', 'passion'],
            'fear': ['fear', 'scared', 'horror', 'terror', 'danger'],
            'joy': ['happy', 'joy', 'laugh', 'smile', 'celebration'],
            'sadness': ['sad', 'cry', 'tears', 'grief', 'loss'],
            'excitement': ['exciting', 'thrill', 'adventure', 'energy']
        }
        
        detected_emotions = []
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        # 시각적 요소 분석
        visual_keywords = {
            'water': ['water', 'rain', 'ocean', 'river', 'lake'],
            'fire': ['fire', 'flame', 'candle', 'fireplace', 'torch'],
            'flowers': ['flowers', 'roses', 'garden', 'bloom', 'petals'],
            'metal': ['metal', 'steel', 'iron', 'gold', 'silver'],
            'wood': ['wood', 'tree', 'oak', 'pine', 'cedar']
        }
        
        visual_elements = []
        for element, keywords in visual_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                visual_elements.append(element)
        
        return {
            'location': detected_location,
            'time_of_day': detected_time,
            'emotions': detected_emotions if detected_emotions else [mood],
            'visual_elements': visual_elements,
            'scene_type': scene_type,
            'description_length': len(description),
            'complexity_score': len(visual_elements) + len(detected_emotions)
        }
    
    def _predict_with_model(self, scene_data: Dict) -> Dict[str, Any]:
        """실제 딥러닝 모델을 사용한 예측"""
        try:
            # 실제 딥러닝 모델 사용
            if hasattr(self, 'movie_ai') and self.movie_ai and self.movie_ai.model:
                # MovieScentAI의 predict_movie_scent 함수 사용
                model_result = self.movie_ai.predict_movie_scent(scene_data)
                
                # 결과를 우리 형식으로 변환
                return {
                    'intensity': model_result['intensity'],
                    'longevity': model_result['longevity'], 
                    'projection': model_result['projection'],
                    'primary_categories': list(model_result['scent_categories'].keys())[:3],
                    'confidence': model_result['confidence'],
                    'scent_categories': model_result['scent_categories'],
                    'recommended_notes': model_result['recommended_notes']
                }
                
            # 모델이 로드되지 않은 경우 기존 규칙 기반 시스템 사용
            else:
                logger.warning("딥러닝 모델이 로드되지 않음. 규칙 기반 시스템 사용")
                return self._rule_based_prediction(scene_data)
            
        except Exception as e:
            logger.error(f"모델 예측 실패: {e}")
            # 예외 발생 시 폴백
            return self._fallback_prediction(scene_data, 5)
    
    def _rule_based_prediction(self, scene_data: Dict) -> Dict[str, Any]:
        """규칙 기반 예측 (모델이 없을 때)"""
        # 원래의 더미 로직을 개선된 규칙 기반으로 변경
        base_intensity = 5.0
        
        # 감정 기반 강도 조정
        emotions = scene_data.get('emotions', [])
        if 'love' in emotions:
            base_intensity += 1.0
        if 'fear' in emotions:
            base_intensity += 2.0
        if 'joy' in emotions:
            base_intensity += 0.5
        if 'sadness' in emotions:
            base_intensity += 1.5
        if 'anger' in emotions:
            base_intensity += 2.5
            
        # 장면 타입별 조정
        scene_type = scene_data.get('scene_type', 'drama')
        if scene_type == 'action':
            base_intensity += 2.0
        elif scene_type == 'horror':
            base_intensity += 3.0
        elif scene_type == 'romantic':
            base_intensity += 0.5
        elif scene_type == 'comedy':
            base_intensity -= 1.0
            
        return {
            'intensity': min(10.0, max(1.0, base_intensity)),
            'longevity': 6.0 + scene_data.get('complexity_score', 0),
            'projection': 5.0 + len(scene_data.get('visual_elements', [])),
            'primary_categories': self._determine_primary_categories(scene_data),
            'confidence': 0.75  # 규칙 기반이므로 약간 낮은 신뢰도
        }
    
    def _fallback_prediction(self, scene_data: Dict, intensity_preference: int) -> Dict[str, Any]:
        """폴백 예측"""
        
        # 규칙 기반 예측
        primary_categories = self._determine_primary_categories(scene_data)
        
        intensity = intensity_preference
        if 'fear' in scene_data['emotions']:
            intensity = min(10, intensity + 3)
        elif 'love' in scene_data['emotions']:
            intensity = max(3, intensity + 1)
        
        longevity = 6
        if scene_data['location'] in ['forest', 'beach']:
            longevity += 1
        
        projection = 5
        if scene_data['scene_type'] in ['action', 'thriller']:
            projection += 2
        
        return {
            'intensity': intensity,
            'longevity': longevity,
            'projection': projection,
            'primary_categories': primary_categories,
            'confidence': 0.7
        }
    
    def _determine_primary_categories(self, scene_data: Dict) -> List[str]:
        """주요 향수 카테고리 결정"""
        categories = []
        
        # 감정 기반
        if 'love' in scene_data['emotions']:
            categories.extend(['floral', 'oriental', 'gourmand'])
        if 'fear' in scene_data['emotions']:
            categories.extend(['smoky', 'earthy', 'metallic'])
        if 'joy' in scene_data['emotions']:
            categories.extend(['citrus', 'fresh', 'fruity'])
        
        # 위치 기반
        location_mapping = {
            'beach': ['aquatic', 'fresh', 'citrus'],
            'forest': ['woody', 'herbal', 'earthy'],
            'city': ['metallic', 'synthetic', 'smoky'],
            'home': ['gourmand', 'floral', 'oriental'],
            'restaurant': ['gourmand', 'spicy', 'herbal']
        }
        
        if scene_data['location'] in location_mapping:
            categories.extend(location_mapping[scene_data['location']])
        
        # 시간 기반
        time_mapping = {
            'morning': ['citrus', 'fresh'],
            'afternoon': ['floral', 'fruity'], 
            'evening': ['oriental', 'woody'],
            'night': ['smoky', 'animalic']
        }
        
        if scene_data['time_of_day'] in time_mapping:
            categories.extend(time_mapping[scene_data['time_of_day']])
        
        # 중복 제거 및 상위 5개 반환
        unique_categories = list(set(categories))
        return unique_categories[:5] if unique_categories else ['floral', 'fresh']
    
    def _recommend_specific_perfumes(self, prediction: Dict, scene_type: str, mood: str) -> Dict[str, Any]:
        """구체적인 향수 제품 추천"""
        
        recommendations = {
            'top_picks': [],
            'alternatives': [],
            'budget_options': [],
            'niche_selections': []
        }
        
        # 강도와 무드에 따른 카테고리 매핑
        intensity = prediction['intensity']
        
        if mood in ['romantic', 'love'] or 'love' in str(prediction.get('emotions', [])):
            category = 'romantic'
        elif intensity >= 7 or scene_type in ['action', 'thriller']:
            category = 'intense'
        elif scene_type in ['comedy', 'light'] or intensity <= 4:
            category = 'fresh'
        else:
            category = 'sophisticated'
        
        # 해당 카테고리의 향수들 선택
        if category in self.perfume_database:
            for brand, perfumes in self.perfume_database[category].items():
                for perfume in perfumes[:2]:  # 브랜드당 최대 2개
                    recommendations['top_picks'].append({
                        'brand': brand,
                        'name': perfume,
                        'category': category,
                        'intensity_match': abs(intensity - 5) <= 2,  # 강도 매치 여부
                        'confidence': prediction['confidence']
                    })
        
        # 대안 추천 (다른 카테고리에서)
        alt_categories = ['romantic', 'intense', 'fresh', 'sophisticated']
        alt_categories.remove(category)
        
        for alt_category in alt_categories[:2]:
            if alt_category in self.perfume_database:
                brand = list(self.perfume_database[alt_category].keys())[0]
                perfume = self.perfume_database[alt_category][brand][0]
                recommendations['alternatives'].append({
                    'brand': brand,
                    'name': perfume,
                    'category': alt_category,
                    'reason': f"Alternative style for {scene_type} scenes"
                })
        
        # 예산 옵션 (가상)
        budget_brands = ['Zara', 'The Body Shop', 'Bath & Body Works']
        for brand in budget_brands[:3]:
            recommendations['budget_options'].append({
                'brand': brand,
                'name': f"{category.title()} Collection",
                'category': category,
                'price_range': 'budget'
            })
        
        # 니치 셀렉션 (가상)
        niche_brands = ['Le Labo', 'Diptyque', 'Byredo', 'Maison Francis Kurkdjian']
        for brand in niche_brands[:2]:
            recommendations['niche_selections'].append({
                'brand': brand,
                'name': f"{category.title()} Essence",
                'category': category,
                'uniqueness': 'high'
            })
        
        return recommendations
    
    def _emergency_fallback(self, description: str, scene_type: str) -> Dict[str, Any]:
        """비상 폴백 추천"""
        return {
            'scene_analysis': {'description': description, 'type': scene_type},
            'scent_profile': {'intensity': 5, 'longevity': 6, 'projection': 5, 'confidence': 0.5},
            'product_recommendations': {
                'top_picks': [
                    {'brand': 'Chanel', 'name': 'No.5', 'category': 'classic'},
                    {'brand': 'Dior', 'name': 'Sauvage', 'category': 'universal'}
                ]
            },
            'meta': {'emergency_mode': True}
        }
    
    def _update_stats(self, response_time: float):
        """성능 통계 업데이트"""
        total = self.stats['total_recommendations']
        current_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = ((current_avg * (total - 1)) + response_time) / total
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        cache_hit_rate = (self.stats['cache_hits'] / max(1, self.stats['total_recommendations'])) * 100
        
        return {
            'total_recommendations': self.stats['total_recommendations'],
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'average_response_time': f"{self.stats['average_response_time']:.3f}초",
            'cache_size': len(self.recommendation_cache),
            'model_loaded': self.model is not None
        }

def demo_real_time_recommendations():
    """실시간 추천 데모"""
    print("=" * 60)
    print("🎬 실시간 영화 장면 향수 추천 시스템 데모")
    print("=" * 60)
    
    # 시스템 초기화
    recommender = RealTimeMovieScentRecommender()
    model_loaded = recommender.load_model_and_preprocessor()
    
    print(f"모델 로드 상태: {'✅ 성공' if model_loaded else '❌ 폴백 모드'}")
    print()
    
    # 테스트 시나리오들
    test_scenarios = [
        {
            'description': "해변가에서 석양을 바라보며 와인을 마시는 로맨틱한 데이트 장면",
            'scene_type': "romantic",
            'mood': "love",
            'intensity_preference': 6
        },
        {
            'description': "어둠 속 폐허에서 괴물과 마주치는 공포스러운 순간",
            'scene_type': "horror", 
            'mood': "fear",
            'intensity_preference': 9
        },
        {
            'description': "파리의 작은 카페에서 크루아상과 커피를 즐기는 평화로운 아침",
            'scene_type': "slice_of_life",
            'mood': "peaceful",
            'intensity_preference': 4
        },
        {
            'description': "고속 추격전 중 폭발이 일어나는 액션 시퀀스",
            'scene_type': "action",
            'mood': "excitement", 
            'intensity_preference': 10
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"[테스트 {i}] {scenario['scene_type'].upper()}")
        print(f"장면: {scenario['description']}")
        print("-" * 50)
        
        # 추천 실행
        recommendation = recommender.recommend_for_scene(
            scenario['description'],
            scenario['scene_type'],
            scenario['mood'], 
            scenario['intensity_preference']
        )
        
        # 결과 출력
        scent = recommendation['scent_profile']
        print(f"🎯 향수 프로필:")
        print(f"   강도: {scent['intensity']:.1f}/10")
        print(f"   지속성: {scent['longevity']:.1f}/10")
        print(f"   투사력: {scent['projection']:.1f}/10")
        print(f"   주요 카테고리: {', '.join(scent['primary_categories'])}")
        
        print(f"🏆 추천 제품:")
        top_picks = recommendation['product_recommendations']['top_picks'][:3]
        for j, pick in enumerate(top_picks, 1):
            print(f"   {j}. {pick['brand']} - {pick['name']}")
        
        meta = recommendation['meta']
        print(f"⚡ 처리 시간: {meta['response_time']:.3f}초")
        print(f"🎯 신뢰도: {meta['confidence']:.1%}")
        print()
    
    # 성능 통계
    stats = recommender.get_performance_stats()
    print("📊 시스템 성능 통계:")
    print(f"   총 추천 수: {stats['total_recommendations']}")
    print(f"   캐시 적중률: {stats['cache_hit_rate']}")
    print(f"   평균 응답 시간: {stats['average_response_time']}")
    print(f"   캐시 크기: {stats['cache_size']}")
    print()
    
    print("✅ 실시간 추천 시스템 데모 완료!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_real_time_recommendations()