#!/usr/bin/env python3
"""
🎬 Ultimate Movie Scent AI System
10만개 데이터로 훈련된 딥러닝 모델을 사용한 최고 성능 영화 향료 시스템
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# 상위 디렉토리 추가
sys.path.append(str(Path(__file__).parent))

try:
    from test_trained_model import TrainedMovieScentPredictor
    from core.movie_capsule_formulator import get_capsule_formulator
    from core.scene_fragrance_recipe import SceneFragranceRecipe
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are available")

class UltimateMovieScentSystem:
    """최고 성능 영화 향료 AI 시스템"""
    
    def __init__(self):
        print("Ultimate Movie Scent AI System 초기화 중...")
        
        # 딥러닝 예측기 로드
        try:
            self.dl_predictor = TrainedMovieScentPredictor()
            self.dl_available = True
            print("딥러닝 모델 로드 완료 (10만개 데이터 훈련)")
        except Exception as e:
            print(f"딥러닝 모델 로드 실패: {e}")
            self.dl_available = False
        
        # 백업 시스템들
        self.recipe_generator = SceneFragranceRecipe()
        try:
            self.capsule_formulator = get_capsule_formulator()
            self.capsule_available = True
            print("캡슐 제조 시스템 로드 완료")
        except:
            self.capsule_available = False
        
        print("백업 레시피 시스템 로드 완료")
        
        # 성능 통계
        self.prediction_stats = {
            'total_predictions': 0,
            'dl_predictions': 0,
            'fallback_predictions': 0,
            'average_confidence': 0.0
        }
    
    def predict_movie_scent(self, scene_description: str, genre: str = "drama", 
                           target_duration: float = 7.0, 
                           output_format: str = "detailed") -> Dict[str, Any]:
        """영화 장면으로부터 최적의 향료 조합 예측"""
        
        start_time = time.time()
        self.prediction_stats['total_predictions'] += 1
        
        result = {
            'scene_description': scene_description,
            'genre': genre,
            'target_duration': target_duration,
            'prediction_method': 'unknown',
            'processing_time': 0.0,
            'system_confidence': 0.0
        }
        
        try:
            if self.dl_available:
                # 1차: 딥러닝 모델 예측 (가장 정확)
                dl_result = self.dl_predictor.predict_scene_fragrance(scene_description, genre)
                
                result.update({
                    'prediction_method': 'deep_learning',
                    'volatility_level': dl_result['predicted_volatility'],
                    'duration_estimate': dl_result['predicted_duration'],
                    'detected_emotions': [e['emotion'] for e in dl_result['predicted_emotions']],
                    'fragrance_notes': dl_result['fragrance_notes'],
                    'system_confidence': dl_result['confidence_score'],
                    'total_materials': dl_result['total_materials'],
                    'recommendation_summary': dl_result['recommendation_summary']
                })
                
                # 상세 정보 추가
                if output_format == "detailed":
                    result['detailed_analysis'] = {
                        'emotion_confidences': dl_result['predicted_emotions'],
                        'material_strengths': self._analyze_material_strengths(dl_result['fragrance_notes']),
                        'scene_complexity': self._calculate_scene_complexity(scene_description),
                        'genre_compatibility': self._check_genre_compatibility(genre, dl_result)
                    }
                
                # 캡슐 제조 정보 추가 (요청 시)
                if self.capsule_available and target_duration <= 10.0:
                    capsule_info = self._generate_capsule_info(scene_description, target_duration)
                    result['capsule_manufacturing'] = capsule_info
                
                self.prediction_stats['dl_predictions'] += 1
                self.prediction_stats['average_confidence'] = (
                    (self.prediction_stats['average_confidence'] * (self.prediction_stats['total_predictions'] - 1) + 
                     dl_result['confidence_score']) / self.prediction_stats['total_predictions']
                )
                
            else:
                # 2차: 백업 시스템 사용
                backup_result = self.recipe_generator.generate_recipe(scene_description)
                
                result.update({
                    'prediction_method': 'rule_based_fallback',
                    'volatility_level': backup_result['volatility_level'],
                    'duration_estimate': backup_result['duration_estimate'],
                    'detected_emotions': backup_result['detected_emotions'],
                    'fragrance_notes': backup_result['fragrance_notes'],
                    'system_confidence': 0.75,  # 규칙 기반은 중간 신뢰도
                    'total_materials': len(backup_result['fragrance_notes']['top_notes']) + 
                                     len(backup_result['fragrance_notes']['middle_notes']) + 
                                     len(backup_result['fragrance_notes']['base_notes']),
                    'recommendation_summary': f"규칙 기반 분석 결과"
                })
                
                self.prediction_stats['fallback_predictions'] += 1
        
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            # 최종 백업: 간단한 기본 레시피
            result.update(self._generate_emergency_recipe(scene_description, genre))
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _analyze_material_strengths(self, fragrance_notes: Dict) -> Dict[str, str]:
        """원료별 강도 분석"""
        strengths = {}
        
        for note_type, notes in fragrance_notes.items():
            for note in notes:
                concentration = note.get('concentration_percent', 0)
                if concentration > 5:
                    strengths[note['name']] = 'strong'
                elif concentration > 2:
                    strengths[note['name']] = 'medium'
                else:
                    strengths[note['name']] = 'subtle'
        
        return strengths
    
    def _calculate_scene_complexity(self, scene_description: str) -> str:
        """장면 복잡도 계산"""
        complexity_indicators = [
            'explosion', 'chase', 'fight', 'battle', 'multiple', 'crowd',
            'action', 'intense', 'dramatic', 'emotional', 'romantic'
        ]
        
        matches = sum(1 for indicator in complexity_indicators 
                     if indicator in scene_description.lower())
        
        if matches >= 3:
            return 'high'
        elif matches >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _check_genre_compatibility(self, genre: str, dl_result: Dict) -> float:
        """장르 호환성 점수"""
        predicted_emotions = [e['emotion'] for e in dl_result['predicted_emotions']]
        
        genre_emotion_map = {
            'action': ['neutral', 'happy'],
            'romantic': ['romantic', 'warm', 'happy'],
            'horror': ['scary', 'mysterious'],
            'drama': ['sad', 'neutral', 'warm'],
            'thriller': ['scary', 'mysterious', 'neutral'],
            'comedy': ['happy', 'neutral'],
            'sci_fi': ['mysterious', 'neutral']
        }
        
        expected_emotions = genre_emotion_map.get(genre, ['neutral'])
        matches = sum(1 for emotion in predicted_emotions if emotion in expected_emotions)
        
        return min(1.0, matches / max(1, len(expected_emotions)))
    
    def _generate_capsule_info(self, scene_description: str, target_duration: float) -> Dict:
        """캡슐 제조 정보 생성"""
        try:
            formula = self.capsule_formulator.formulate_capsule(scene_description, target_duration)
            
            return {
                'available': True,
                'estimated_cost': f"${formula.estimated_cost_per_unit:.4f}/개",
                'diffusion_control': formula.diffusion_control,
                'encapsulation_method': formula.encapsulation_method,
                'activation_mechanism': formula.activation_mechanism,
                'raw_material_count': len(formula.raw_materials),
                'production_feasible': True
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e),
                'production_feasible': False
            }
    
    def _generate_emergency_recipe(self, scene_description: str, genre: str) -> Dict:
        """비상용 기본 레시피"""
        return {
            'prediction_method': 'emergency_fallback',
            'volatility_level': 'medium_volatility',
            'duration_estimate': '2-3분 (기본값)',
            'detected_emotions': ['neutral'],
            'fragrance_notes': {
                'top_notes': [{'name': 'bergamot', 'concentration_percent': 3.0}],
                'middle_notes': [{'name': 'lavender', 'concentration_percent': 5.0}],
                'base_notes': [{'name': 'cedar', 'concentration_percent': 4.0}]
            },
            'system_confidence': 0.3,
            'total_materials': 3,
            'recommendation_summary': '기본 안전 레시피 (시스템 오류 시)'
        }
    
    def batch_predict_scenes(self, scenes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """여러 장면을 배치로 예측"""
        print(f"배치 예측 시작: {len(scenes)}개 장면")
        results = []
        
        for i, scene_info in enumerate(scenes, 1):
            print(f"  처리중 {i}/{len(scenes)}: {scene_info.get('description', '')[:50]}...")
            
            result = self.predict_movie_scent(
                scene_info['description'],
                scene_info.get('genre', 'drama'),
                scene_info.get('target_duration', 7.0)
            )
            results.append(result)
        
        print(f"배치 예측 완료!")
        return results
    
    def get_system_performance(self) -> Dict[str, Any]:
        """시스템 성능 통계"""
        total = self.prediction_stats['total_predictions']
        
        performance = {
            'total_predictions': total,
            'deep_learning_usage': f"{(self.prediction_stats['dl_predictions']/max(1,total)*100):.1f}%",
            'fallback_usage': f"{(self.prediction_stats['fallback_predictions']/max(1,total)*100):.1f}%",
            'average_confidence': f"{self.prediction_stats['average_confidence']:.3f}",
            'system_status': {
                'deep_learning_model': 'available' if self.dl_available else 'unavailable',
                'capsule_manufacturing': 'available' if self.capsule_available else 'unavailable',
                'rule_based_backup': 'available'
            }
        }
        
        return performance
    
    def save_prediction_results(self, results: List[Dict], output_path: str = "prediction_results.json"):
        """예측 결과 저장"""
        output_file = Path(output_path)
        
        # JSON 직렬화를 위해 numpy 타입들을 Python 기본 타입으로 변환
        def convert_numpy_types(obj):
            import numpy as np
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy_types(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"예측 결과 저장됨: {output_file.absolute()}")

def demo_ultimate_system():
    """궁극적 시스템 데모"""
    print("=" * 70)
    print("ULTIMATE MOVIE SCENT AI SYSTEM")
    print("10만개 영화 장면 데이터로 훈련된 딥러닝 모델 기반")
    print("=" * 70)
    
    # 시스템 초기화
    system = UltimateMovieScentSystem()
    
    print("\n고급 영화 장면 테스트")
    print("-" * 50)
    
    # 복잡한 영화 장면들 테스트
    complex_scenes = [
        {
            "description": "어벤져스 엔드게임 최종 전투: 토니 스타크가 인피니티 스톤을 들고 타노스와 대결하는 감동적이면서도 웅장한 순간",
            "genre": "action",
            "target_duration": 8.0
        },
        {
            "description": "타이타닉 침몰 장면: 로즈와 잭이 차가운 바닷물 속에서 마지막 키스를 나누는 비극적 로맨스",
            "genre": "romantic", 
            "target_duration": 12.0
        },
        {
            "description": "기생충 반지하 침수 장면: 폭우로 인해 집이 물에 잠기면서 가족의 절망이 극에 달하는 순간",
            "genre": "drama",
            "target_duration": 15.0
        }
    ]
    
    # 배치 예측
    results = system.batch_predict_scenes(complex_scenes)
    
    # 결과 출력
    for i, result in enumerate(results, 1):
        print(f"\n[예측 {i}] {result['genre'].upper()} - {result['prediction_method'].upper()}")
        print(f"장면: {result['scene_description'][:80]}...")
        print(f"처리 시간: {result['processing_time']:.3f}초")
        print(f"신뢰도: {result['system_confidence']:.2f}")
        print(f"총 원료: {result['total_materials']}종")
        print(f"휘발성: {result['volatility_level']}")
        print(f"지속시간: {result['duration_estimate']}")
        
        if result.get('detailed_analysis'):
            analysis = result['detailed_analysis']
            print(f"장면 복잡도: {analysis['scene_complexity']}")
            print(f"장르 호환성: {analysis['genre_compatibility']:.2f}")
        
        if result.get('capsule_manufacturing', {}).get('available'):
            capsule = result['capsule_manufacturing']
            print(f"캡슐 제조 가능: {capsule['estimated_cost']}")
        
        print(f"요약: {result['recommendation_summary']}")
    
    # 성능 통계
    print(f"\n시스템 성능 통계")
    print("-" * 30)
    performance = system.get_system_performance()
    for key, value in performance.items():
        if key != 'system_status':
            print(f"{key}: {value}")
    
    print(f"\n시스템 상태")
    print("-" * 20)
    for component, status in performance['system_status'].items():
        print(f"{component}: {status}")
    
    # 결과 저장
    system.save_prediction_results(results, "ultimate_prediction_results.json")
    
    print(f"\nUltimate Movie Scent AI 데모 완료!")
    print(f"딥러닝 모델을 통해 영화 장면별 최적화된 향료 조합을 제공합니다.")

if __name__ == "__main__":
    demo_ultimate_system()