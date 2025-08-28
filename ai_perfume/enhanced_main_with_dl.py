#!/usr/bin/env python3
"""
딥러닝이 통합된 최신 향수 AI 시스템
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
import time

# 현재 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent))

# 기존 시스템 임포트
from models.text_analyzer import TextAnalyzer
from models.advanced_recipe_generator import AdvancedRecipeGenerator
from core.database import Session, Recipe, Feedback, init_db, get_db_session

# 딥러닝 통합 모듈 임포트
try:
    from core.deep_learning_integration import DeepLearningPerfumePredictor, EnhancedPerfumeGenerator
    DL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: 딥러닝 모듈 사용 불가: {e}")
    DL_AVAILABLE = False

# 기타 고급 시스템들
try:
    from core.rag_system import FragranceRAGSystem
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG system not available: {e}")
    RAG_AVAILABLE = False

try:
    from core.korean_llm_integration import KoreanFragranceLLMSystem
    KOREAN_LLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Korean LLM system not available: {e}")
    KOREAN_LLM_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeepLearningEnhancedPerfumeAI:
    """딥러닝이 통합된 차세대 향수 AI 시스템"""
    
    def __init__(self):
        logger.info("🚀 딥러닝 통합 향수 AI 시스템 초기화 중...")
        
        # 기본 시스템들
        self.text_analyzer = TextAnalyzer()
        self.advanced_generator = AdvancedRecipeGenerator()
        
        # 딥러닝 시스템 초기화
        self.dl_predictor = None
        self.enhanced_generator = None
        self._initialize_deep_learning()
        
        # 선택적 고급 시스템들
        self.rag_system = None
        self.korean_llm_system = None
        self._initialize_optional_systems()
        
        # 시스템 성능 통계
        self.performance_stats = {
            'total_requests': 0,
            'successful_generations': 0,
            'dl_enhanced_generations': 0,
            'average_response_time': 0.0,
            'system_components_active': self._get_active_components()
        }
        
        logger.info("✅ 딥러닝 통합 향수 AI 시스템 초기화 완료!")
        self._print_system_status()
    
    def _initialize_deep_learning(self):
        """딥러닝 시스템 초기화"""
        if not DL_AVAILABLE:
            logger.warning("❌ 딥러닝 모듈 사용 불가")
            return
        
        try:
            # 딥러닝 모델 경로
            model_path = "models/perfume_dl_model.pth"
            preprocessor_path = "data/processed/preprocessor_tools.pkl"  
            metadata_path = "data/processed/metadata.json"
            
            # 파일 존재 확인
            if not all(Path(p).exists() for p in [model_path, preprocessor_path, metadata_path]):
                logger.warning("⚠️ 딥러닝 모델 파일들이 아직 준비되지 않음 - 훈련 완료 후 재시작 필요")
                return
            
            # 딥러닝 예측기 초기화
            self.dl_predictor = DeepLearningPerfumePredictor(
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                metadata_path=metadata_path
            )
            
            # 향상된 생성기 초기화
            self.enhanced_generator = EnhancedPerfumeGenerator(self.dl_predictor)
            
            logger.info("✅ 딥러닝 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 딥러닝 시스템 초기화 실패: {e}")
    
    def _initialize_optional_systems(self):
        """선택적 고급 시스템들 초기화"""
        
        # RAG 시스템
        if RAG_AVAILABLE:
            try:
                self.rag_system = FragranceRAGSystem()
                logger.info("✅ RAG 시스템 초기화 완료")
            except Exception as e:
                logger.error(f"❌ RAG 시스템 초기화 실패: {e}")
        
        # 한국어 LLM 시스템
        if KOREAN_LLM_AVAILABLE:
            try:
                self.korean_llm_system = KoreanFragranceLLMSystem()
                logger.info("✅ 한국어 LLM 시스템 초기화 완료")
            except Exception as e:
                logger.error(f"❌ 한국어 LLM 시스템 초기화 실패: {e}")
    
    def _get_active_components(self) -> List[str]:
        """활성화된 시스템 컴포넌트 목록"""
        active = ['기본_텍스트_분석', '고급_레시피_생성']
        
        if self.dl_predictor:
            active.append('딥러닝_예측')
        if self.enhanced_generator:
            active.append('딥러닝_통합_생성')
        if self.rag_system:
            active.append('RAG_지식베이스')
        if self.korean_llm_system:
            active.append('한국어_LLM')
        
        return active
    
    def _print_system_status(self):
        """시스템 상태 출력"""
        print("\n" + "="*80)
        print("🤖 DEEP LEARNING ENHANCED PERFUME AI SYSTEM")
        print("="*80)
        
        active_components = self._get_active_components()
        print(f"📊 활성 컴포넌트: {len(active_components)}개")
        for i, component in enumerate(active_components, 1):
            status = "✅" if component != "딥러닝_예측" or self.dl_predictor else "⚠️"
            print(f"   {i}. {status} {component}")
        
        # 시스템 특성
        features = [
            "🧠 Transformer 기반 레시피 생성",
            "🤖 딥러닝 평점/성별 예측" if self.dl_predictor else "❌ 딥러닝 시스템 대기중",
            "📈 ML 기반 노트 추천" if self.enhanced_generator else "❌ 향상된 생성기 대기중",
            "🔍 RAG 전문 지식 검색" if self.rag_system else "❌ RAG 시스템 비활성",
            "🇰🇷 한국어 LLM 통합" if self.korean_llm_system else "❌ 한국어 LLM 비활성",
            "💾 지속적 학습 및 피드백",
            "🎨 어텐션 메커니즘 향료 최적화"
        ]
        
        print(f"\n🚀 시스템 기능:")
        for feature in features:
            print(f"   {feature}")
        
        if self.dl_predictor:
            dl_info = self.dl_predictor.get_model_info()
            print(f"\n🤖 딥러닝 모델 정보:")
            print(f"   - 모델 상태: {'활성' if dl_info['model_available'] else '비활성'}")
            print(f"   - 입력 차원: {dl_info['metadata'].get('feature_dim', 'N/A')}")
            print(f"   - 출력 차원: {dl_info['metadata'].get('target_dim', 'N/A')}")
            print(f"   - 훈련 샘플: {dl_info['metadata'].get('num_samples', 'N/A')}")
        
        print("="*80)
    
    def generate_perfume_with_deep_learning(
        self,
        text: str,
        user_preferences: Optional[Dict] = None,
        use_rag: bool = True,
        generate_korean_content: bool = True
    ) -> Dict[str, Any]:
        """딥러닝이 통합된 향수 생성"""
        
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        logger.info(f"🎯 딥러닝 통합 향수 생성 요청: '{text[:50]}...'")
        
        try:
            # 1. 기본 텍스트 분석
            logger.info("📝 텍스트 분석 중...")
            analysis = self.text_analyzer.analyze(text)
            
            # 2. 딥러닝 시스템 사용 가능한 경우
            if self.enhanced_generator:
                logger.info("🤖 딥러닝 통합 레시피 생성 중...")
                recipe = self.enhanced_generator.generate_enhanced_recipe(
                    text, analysis, user_preferences
                )
                self.performance_stats['dl_enhanced_generations'] += 1
            else:
                logger.info("⚡ 기본 Transformer 레시피 생성 중...")
                recipe = self._generate_basic_recipe(analysis)
            
            # 3. RAG 시스템으로 레시피 강화 (선택사항)
            if use_rag and self.rag_system:
                logger.info("🔍 RAG 지식베이스로 레시피 강화 중...")
                try:
                    enhanced_recipe = self.rag_system.enhance_recipe_with_context(
                        recipe, f"감정: {list(analysis['emotions'].keys())}"
                    )
                    recipe.update(enhanced_recipe)
                except Exception as e:
                    logger.warning(f"RAG 강화 실패: {e}")
            
            # 4. 한국어 콘텐츠 생성 (선택사항)
            korean_content = {}
            if generate_korean_content and self.korean_llm_system:
                logger.info("🇰🇷 한국어 콘텐츠 생성 중...")
                try:
                    recipe_data = {
                        'top_notes': recipe.get('top_notes', []),
                        'middle_notes': recipe.get('middle_notes', []),
                        'base_notes': recipe.get('base_notes', []),
                        'emotions': list(analysis['emotions'].keys())[:3],
                        'intensity': recipe.get('intensity', 5.0)
                    }
                    korean_content = self.korean_llm_system.generate_korean_fragrance_content(
                        recipe_data,
                        content_types=['name', 'description', 'story'],
                        model_preference='kogpt2'
                    )
                except Exception as e:
                    logger.warning(f"한국어 콘텐츠 생성 실패: {e}")
            
            # 5. 최종 결과 통합
            final_result = self._combine_results(recipe, korean_content, analysis, text)
            
            # 6. 데이터베이스 저장
            db_recipe_id = self._save_to_database(final_result, text)
            final_result['recipe_id'] = db_recipe_id
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, success=True)
            
            final_result['processing_time'] = processing_time
            final_result['system_info'] = {
                'deep_learning_enhanced': bool(self.enhanced_generator),
                'rag_enhanced': use_rag and self.rag_system,
                'korean_content_generated': bool(korean_content),
                'generation_quality': self._assess_generation_quality(final_result),
                'active_components': len(self._get_active_components())
            }
            
            logger.info(f"✅ 딥러닝 통합 향수 생성 완료 ({processing_time:.2f}초)")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 향수 생성 실패: {e}")
            self._update_performance_stats(time.time() - start_time, success=False)
            return self._generate_fallback_recipe(text)
    
    def _generate_basic_recipe(self, analysis: Dict) -> Dict[str, Any]:
        """기본 레시피 생성 (딥러닝 없이)"""
        import torch
        
        emotion_scores = torch.tensor(
            [list(analysis['emotions'].values())],
            dtype=torch.float32
        )
        
        return self.advanced_generator.generate_recipe(
            analysis['embeddings'],
            emotion_scores,
            temperature=0.8,
            top_k=5
        )
    
    def _combine_results(self, recipe: Dict, korean_content: Dict, 
                        analysis: Dict, original_text: str) -> Dict[str, Any]:
        """모든 결과를 통합"""
        
        combined_result = {
            # 기본 정보
            'name': korean_content.get('name', recipe.get('name', '향수')),
            'description': korean_content.get('description', recipe.get('description', '특별한 향수입니다.')),
            
            # 레시피 구성
            'top_notes': recipe.get('top_notes', []),
            'middle_notes': recipe.get('middle_notes', []),
            'base_notes': recipe.get('base_notes', []),
            'intensity': recipe.get('intensity', 5.0),
            
            # 품질 점수들
            'composition_harmony': recipe.get('composition_harmony', 0.7),
            'confidence_scores': recipe.get('confidence_scores', {}),
            
            # 딥러닝 예측 정보 (있는 경우)
            'ml_predictions': {
                'predicted_rating': recipe.get('predicted_rating'),
                'predicted_gender': recipe.get('predicted_gender'),
                'gender_probabilities': recipe.get('gender_probabilities', {}),
                'ml_confidence': recipe.get('ml_confidence'),
                'ml_enhanced': recipe.get('ml_enhanced', False)
            },
            
            # 한국어 콘텐츠
            'korean_content': {
                'emotional_story': korean_content.get('emotional_story', ''),
                'usage_recommendation': korean_content.get('usage_recommendation', ''),
                'cultural_context': korean_content.get('cultural_context', '')
            },
            
            # AI 시스템 정보
            'ai_insights': {
                'original_emotions': analysis['emotions'],
                'original_keywords': analysis['keywords'],
                'dl_enhanced': bool(self.enhanced_generator and recipe.get('ml_enhanced')),
                'korean_llm_generated': bool(korean_content)
            }
        }
        
        return combined_result
    
    def _assess_generation_quality(self, result: Dict) -> str:
        """생성 품질 평가"""
        quality_indicators = [
            len(result.get('top_notes', [])) >= 2,
            len(result.get('middle_notes', [])) >= 2,
            len(result.get('base_notes', [])) >= 1,
            result.get('composition_harmony', 0) > 0.5,
            bool(result.get('description')),
            result.get('ml_predictions', {}).get('ml_enhanced', False),
            result.get('ml_predictions', {}).get('ml_confidence', 0) > 0.6
        ]
        
        quality_score = sum(quality_indicators) / len(quality_indicators)
        
        if quality_score >= 0.8:
            return "최고급"
        elif quality_score >= 0.65:
            return "고품질"
        elif quality_score >= 0.5:
            return "양호"
        else:
            return "기본"
    
    def _generate_fallback_recipe(self, text: str) -> Dict[str, Any]:
        """최종 폴백 레시피"""
        return {
            'name': '기본 향수',
            'description': f'"{text}"에서 영감을 받은 향수입니다.',
            'top_notes': ['bergamot', 'lemon'],
            'middle_notes': ['rose', 'jasmine'],
            'base_notes': ['musk', 'cedar'],
            'intensity': 5.0,
            'system_info': {
                'fallback_mode': True,
                'generation_quality': '기본',
                'deep_learning_enhanced': False
            }
        }
    
    def _save_to_database(self, recipe: Dict, original_text: str) -> int:
        """데이터베이스에 레시피 저장"""
        try:
            with get_db_session() as session:
                db_recipe = Recipe(
                    name=recipe.get('name', '향수'),
                    description=recipe.get('description', ''),
                    top_notes=','.join(recipe.get('top_notes', [])),
                    middle_notes=','.join(recipe.get('middle_notes', [])),
                    base_notes=','.join(recipe.get('base_notes', [])),
                    intensity=recipe.get('intensity', 5.0)
                )
                session.add(db_recipe)
                session.commit()
                return db_recipe.id
        except Exception as e:
            logger.error(f"데이터베이스 저장 실패: {e}")
            return 0
    
    def _update_performance_stats(self, processing_time: float, success: bool):
        """성능 통계 업데이트"""
        if success:
            self.performance_stats['successful_generations'] += 1
        
        total = self.performance_stats['total_requests']
        current_avg = self.performance_stats['average_response_time']
        self.performance_stats['average_response_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """시스템 성능 리포트"""
        
        report = {
            'performance_stats': self.performance_stats.copy(),
            'deep_learning_stats': {
                'dl_available': bool(self.dl_predictor),
                'dl_enhanced_rate': (
                    self.performance_stats['dl_enhanced_generations'] / 
                    max(1, self.performance_stats['total_requests'])
                )
            },
            'system_components': {
                'total_active_components': len(self._get_active_components()),
                'components': self._get_active_components()
            }
        }
        
        # 딥러닝 모델 정보 추가
        if self.dl_predictor:
            report['deep_learning_model_info'] = self.dl_predictor.get_model_info()
        
        # 전체 시스템 건강성
        success_rate = (
            self.performance_stats['successful_generations'] / 
            max(1, self.performance_stats['total_requests'])
        )
        
        report['system_health'] = {
            'success_rate': success_rate,
            'average_response_time': self.performance_stats['average_response_time'],
            'status': 'excellent' if success_rate > 0.95 else 'good' if success_rate > 0.8 else 'needs_attention'
        }
        
        return report

def run_enhanced_demo():
    """딥러닝 통합 시스템 데모"""
    
    print("🤖 딥러닝 통합 향수 AI 시스템 데모")
    print("=" * 80)
    
    # 데이터베이스 초기화
    init_db()
    
    # 시스템 초기화
    perfume_ai = DeepLearningEnhancedPerfumeAI()
    
    # 테스트 시나리오들
    test_scenarios = [
        {
            'text': "봄날 아침 정원에서 피어나는 장미와 함께 느끼는 평온하고 행복한 순간",
            'preferences': {'intensity': 6, 'gender': 'women', 'season': 'spring'},
            'description': "🌸 봄날 장미 정원 시나리오"
        },
        {
            'text': "겨울 저녁 벽난로 앞에서 위스키를 마시며 느끼는 따뜻하고 깊이 있는 감정",
            'preferences': {'intensity': 8, 'gender': 'men', 'season': 'winter'},
            'description': "🔥 겨울 벽난로 시나리오"
        },
        {
            'text': "여름 바다에서 파도 소리를 들으며 느끼는 자유롭고 상쾌한 기분",
            'preferences': {'intensity': 5, 'gender': 'unisex', 'season': 'summer'},
            'description': "🌊 여름 바다 시나리오"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"🎯 테스트 {i}: {scenario['description']}")
        print(f"📝 입력: {scenario['text']}")
        print(f"👤 선호도: {scenario['preferences']}")
        print("-" * 60)
        
        # 딥러닝 통합 레시피 생성
        result = perfume_ai.generate_perfume_with_deep_learning(
            text=scenario['text'],
            user_preferences=scenario['preferences'],
            use_rag=True,
            generate_korean_content=True
        )
        
        # 결과 출력
        print(f"✨ 향수명: {result['name']}")
        print(f"📖 설명: {result['description']}")
        print(f"🎼 구성:")
        print(f"   • 탑 노트: {', '.join(result['top_notes'])}")
        print(f"   • 미들 노트: {', '.join(result['middle_notes'])}")
        print(f"   • 베이스 노트: {', '.join(result['base_notes'])}")
        print(f"💪 강도: {result['intensity']:.1f}/10")
        
        # 딥러닝 예측 정보
        ml_pred = result.get('ml_predictions', {})
        if ml_pred.get('ml_enhanced'):
            print(f"🤖 AI 예측 평점: {ml_pred.get('predicted_rating', 'N/A'):.2f}/5.0")
            print(f"👥 예측 성별: {ml_pred.get('predicted_gender', 'N/A')}")
            print(f"🎯 ML 신뢰도: {ml_pred.get('ml_confidence', 0):.1%}")
        
        # 시스템 정보
        system_info = result.get('system_info', {})
        print(f"🔧 딥러닝 강화: {'✅' if system_info.get('deep_learning_enhanced') else '❌'}")
        print(f"⏱️ 처리 시간: {result.get('processing_time', 0):.2f}초")
        print(f"⭐ 생성 품질: {system_info.get('generation_quality', 'unknown')}")
    
    # 시스템 성능 리포트
    print(f"\n{'='*80}")
    print("📊 시스템 성능 리포트")
    print("=" * 80)
    
    performance_report = perfume_ai.get_system_performance_report()
    
    print(f"📈 총 요청 수: {performance_report['performance_stats']['total_requests']}")
    print(f"✅ 성공 생성: {performance_report['performance_stats']['successful_generations']}")
    print(f"🤖 딥러닝 강화율: {performance_report['deep_learning_stats']['dl_enhanced_rate']:.1%}")
    print(f"⚡ 평균 응답 시간: {performance_report['performance_stats']['average_response_time']:.2f}초")
    print(f"🏥 시스템 상태: {performance_report['system_health']['status']}")
    print(f"💯 성공률: {performance_report['system_health']['success_rate']:.1%}")
    
    active_components = performance_report['system_components']['components']
    print(f"🔧 활성 컴포넌트 ({len(active_components)}개):")
    for component in active_components:
        print(f"   • {component}")
    
    print("\n🎉 딥러닝 통합 향수 AI 시스템 데모 완료!")

if __name__ == "__main__":
    try:
        run_enhanced_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  데모가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 데모 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()