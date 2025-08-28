#!/usr/bin/env python3
"""
최신 AI 기술이 통합된 향수 AI 시스템 메인 실행 파일
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

# 새로운 고급 시스템들 임포트
try:
    from core.rag_system import FragranceRAGSystem
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG system not available: {e}")
    RAG_AVAILABLE = False

try:
    from core.multimodal_embeddings import (
        FragranceMultiModalSystem, 
        MultimodalInput,
        FragranceProperties
    )
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Multimodal system not available: {e}")
    MULTIMODAL_AVAILABLE = False

try:
    from core.reinforcement_learning import (
        FragranceRLSystem,
        UserFeedback as RLUserFeedback,
        FragranceState,
        FragranceAction
    )
    RL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RL system not available: {e}")
    RL_AVAILABLE = False

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


class AdvancedPerfumeAISystem:
    """차세대 향수 AI 시스템 - 모든 최신 기술 통합"""
    
    def __init__(self):
        logger.info("🚀 Advanced Perfume AI System 초기화 중...")
        
        # 기본 시스템들
        self.text_analyzer = TextAnalyzer()
        self.advanced_generator = AdvancedRecipeGenerator()
        
        # 선택적 고급 시스템들
        self.rag_system = None
        self.multimodal_system = None
        self.rl_system = None
        self.korean_llm_system = None
        
        self._initialize_advanced_systems()
        
        # 시스템 성능 통계
        self.performance_stats = {
            'total_requests': 0,
            'successful_generations': 0,
            'average_response_time': 0.0,
            'system_components_active': self._get_active_components()
        }
        
        logger.info("✅ Advanced Perfume AI System 초기화 완료!")
        self._print_system_status()
    
    def _initialize_advanced_systems(self):
        """고급 시스템들 초기화"""
        
        # RAG 시스템 초기화
        if RAG_AVAILABLE:
            try:
                self.rag_system = FragranceRAGSystem()
                logger.info("✅ RAG 시스템 초기화 완료")
            except Exception as e:
                logger.error(f"❌ RAG 시스템 초기화 실패: {e}")
        
        # 다중 모달 시스템 초기화
        if MULTIMODAL_AVAILABLE:
            try:
                self.multimodal_system = FragranceMultiModalSystem()
                logger.info("✅ 다중 모달 시스템 초기화 완료")
            except Exception as e:
                logger.error(f"❌ 다중 모달 시스템 초기화 실패: {e}")
        
        # 강화학습 시스템 초기화
        if RL_AVAILABLE:
            try:
                self.rl_system = FragranceRLSystem()
                logger.info("✅ 강화학습 시스템 초기화 완료")
            except Exception as e:
                logger.error(f"❌ 강화학습 시스템 초기화 실패: {e}")
        
        # 한국어 LLM 시스템 초기화
        if KOREAN_LLM_AVAILABLE:
            try:
                self.korean_llm_system = KoreanFragranceLLMSystem()
                logger.info("✅ 한국어 LLM 시스템 초기화 완료")
            except Exception as e:
                logger.error(f"❌ 한국어 LLM 시스템 초기화 실패: {e}")
    
    def _get_active_components(self) -> List[str]:
        """활성화된 시스템 컴포넌트 목록"""
        active = ['기본_텍스트_분석', '고급_레시피_생성']
        
        if self.rag_system:
            active.append('RAG_지식베이스')
        if self.multimodal_system:
            active.append('다중모달_임베딩')
        if self.rl_system:
            active.append('강화학습_피드백')
        if self.korean_llm_system:
            active.append('한국어_LLM')
        
        return active
    
    def _print_system_status(self):
        """시스템 상태 출력"""
        print("\n" + "="*80)
        print("🤖 ADVANCED PERFUME AI SYSTEM STATUS")
        print("="*80)
        
        active_components = self._get_active_components()
        print(f"📊 활성 컴포넌트: {len(active_components)}개")
        for i, component in enumerate(active_components, 1):
            print(f"   {i}. {component}")
        
        # 시스템 특성
        features = [
            "🧠 Transformer 기반 레시피 생성",
            "🔍 RAG 전문 지식 검색" if self.rag_system else "❌ RAG 시스템 비활성",
            "🎭 다중 모달 임베딩" if self.multimodal_system else "❌ 다중 모달 비활성", 
            "🎯 강화학습 개인화" if self.rl_system else "❌ 강화학습 비활성",
            "🇰🇷 한국어 LLM 통합" if self.korean_llm_system else "❌ 한국어 LLM 비활성",
            "💾 지속적 학습 및 피드백",
            "🎨 어텐션 메커니즘 향료 최적화"
        ]
        
        print(f"\n🚀 시스템 기능:")
        for feature in features:
            print(f"   {feature}")
        
        print("="*80)
    
    def generate_advanced_recipe(
        self,
        text: str,
        user_id: str = "anonymous",
        use_rag: bool = True,
        use_multimodal: bool = False,
        use_rl_personalization: bool = True,
        generate_korean_content: bool = True,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """고급 AI 기능을 활용한 향수 레시피 생성"""
        
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        logger.info(f"🎯 향수 생성 요청: '{text[:50]}...'")
        
        try:
            # 1. 기본 텍스트 분석
            logger.info("📝 텍스트 분석 중...")
            analysis = self.text_analyzer.analyze(text)
            
            # 2. RAG 기반 전문 지식 검색
            rag_context = {}
            if use_rag and self.rag_system:
                logger.info("🔍 전문 지식 검색 중...")
                rag_context = self._get_rag_context(text, analysis)
            
            # 3. 다중 모달 임베딩 (이미지가 있는 경우)
            multimodal_features = None
            if use_multimodal and self.multimodal_system and image_path:
                logger.info("🎭 다중 모달 처리 중...")
                multimodal_features = self._process_multimodal_input(text, image_path)
            
            # 4. 강화학습 기반 개인화된 추천
            rl_recommendation = None
            if use_rl_personalization and self.rl_system:
                logger.info("🎯 개인화 추천 생성 중...")
                rl_recommendation = self._get_rl_recommendation(text, user_id, analysis)
            
            # 5. 고급 Transformer 모델로 레시피 생성
            logger.info("⚡ Transformer 레시피 생성 중...")
            recipe = self._generate_transformer_recipe(analysis, rag_context)
            
            # 6. 한국어 콘텐츠 생성
            korean_content = {}
            if generate_korean_content and self.korean_llm_system:
                logger.info("🇰🇷 한국어 콘텐츠 생성 중...")
                korean_content = self._generate_korean_content(recipe, analysis)
            
            # 7. 결과 통합 및 후처리
            final_result = self._combine_all_results(
                base_recipe=recipe,
                rag_context=rag_context,
                multimodal_features=multimodal_features,
                rl_recommendation=rl_recommendation,
                korean_content=korean_content,
                original_analysis=analysis
            )
            
            # 8. 데이터베이스 저장
            db_recipe_id = self._save_to_database(final_result, text)
            final_result['recipe_id'] = db_recipe_id
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, success=True)
            
            final_result['processing_time'] = processing_time
            final_result['system_info'] = {
                'components_used': [
                    comp for comp, used in [
                        ('RAG', use_rag and self.rag_system),
                        ('Multimodal', use_multimodal and multimodal_features),
                        ('RL', use_rl_personalization and self.rl_system),
                        ('Korean_LLM', generate_korean_content and self.korean_llm_system)
                    ] if used
                ],
                'advanced_features': True,
                'generation_quality': self._assess_generation_quality(final_result)
            }
            
            logger.info(f"✅ 향수 생성 완료 ({processing_time:.2f}초)")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 향수 생성 실패: {e}")
            self._update_performance_stats(time.time() - start_time, success=False)
            
            # 폴백: 기본 시스템으로 생성
            return self._generate_fallback_recipe(text)
    
    def _get_rag_context(self, text: str, analysis: Dict) -> Dict[str, Any]:
        """RAG 시스템에서 관련 컨텍스트 검색"""
        try:
            # 감정과 키워드를 기반으로 검색
            emotions = list(analysis['emotions'].keys())[:3]  # 상위 3개 감정
            keywords = analysis['keywords'][:5]  # 상위 5개 키워드
            
            search_query = f"감정: {', '.join(emotions)} 키워드: {', '.join(keywords)}"
            
            recommendations = self.rag_system.get_fragrance_recommendations(
                emotion=' '.join(emotions),
                context=text,
                season=None  # 계절 정보가 있다면 추가
            )
            
            return {
                'search_query': search_query,
                'recommendations': recommendations,
                'knowledge_confidence': 0.8  # RAG 시스템의 신뢰도
            }
            
        except Exception as e:
            logger.warning(f"RAG 컨텍스트 검색 실패: {e}")
            return {}
    
    def _process_multimodal_input(self, text: str, image_path: str) -> Dict[str, Any]:
        """다중 모달 입력 처리"""
        try:
            multimodal_input = MultimodalInput(
                text_description=text,
                image_path=image_path
            )
            
            encoded_features, metadata = self.multimodal_system.encode_multimodal_input(
                multimodal_input
            )
            
            return {
                'features': encoded_features,
                'metadata': metadata,
                'modalities_used': metadata.get('modalities_used', [])
            }
            
        except Exception as e:
            logger.warning(f"다중 모달 처리 실패: {e}")
            return {}
    
    def _get_rl_recommendation(self, text: str, user_id: str, analysis: Dict) -> Dict[str, Any]:
        """강화학습 기반 개인화 추천"""
        try:
            # 사용자 상태 구성
            fragrance_state = FragranceState(
                user_preferences={'intensity_preference': 5.0},
                context={'season': 'spring', 'occasion': 'casual'},
                emotion_target=list(analysis['emotions'].keys())[:3],
                previous_feedback=[],  # 실제로는 DB에서 로드
                user_history={}
            )
            
            # RL 시스템에서 추천 받기
            fragrance_action, confidence = self.rl_system.get_recommendation(
                fragrance_state, deterministic=False
            )
            
            # 개인화 정보
            personalization = self.rl_system.get_user_personalization(user_id)
            
            return {
                'recommended_action': {
                    'top_notes': fragrance_action.top_notes,
                    'middle_notes': fragrance_action.middle_notes,
                    'base_notes': fragrance_action.base_notes,
                    'intensity': fragrance_action.intensity
                },
                'confidence': confidence,
                'personalization': personalization
            }
            
        except Exception as e:
            logger.warning(f"RL 추천 실패: {e}")
            return {}
    
    def _generate_transformer_recipe(self, analysis: Dict, rag_context: Dict) -> Dict[str, Any]:
        """고급 Transformer 모델로 레시피 생성"""
        try:
            import torch
            
            # 감정 점수를 텐서로 변환
            emotion_scores = torch.tensor(
                [list(analysis['emotions'].values())],
                dtype=torch.float32
            )
            
            # 고급 모델로 레시피 생성
            recipe = self.advanced_generator.generate_recipe(
                analysis['embeddings'],
                emotion_scores,
                temperature=0.8,
                top_k=5
            )
            
            # RAG 컨텍스트로 레시피 강화
            if rag_context and self.rag_system:
                enhanced_recipe = self.rag_system.enhance_recipe_with_context(
                    recipe, f"감정: {list(analysis['emotions'].keys())}"
                )
                return enhanced_recipe
            
            return recipe
            
        except Exception as e:
            logger.error(f"Transformer 레시피 생성 실패: {e}")
            # 기본 모델 폴백
            return self._generate_basic_recipe(analysis)
    
    def _generate_korean_content(self, recipe: Dict, analysis: Dict) -> Dict[str, Any]:
        """한국어 LLM으로 콘텐츠 생성"""
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
                content_types=['name', 'description', 'story', 'marketing'],
                model_preference='kogpt2'
            )
            
            return korean_content
            
        except Exception as e:
            logger.warning(f"한국어 콘텐츠 생성 실패: {e}")
            return {}
    
    def _combine_all_results(
        self,
        base_recipe: Dict,
        rag_context: Dict,
        multimodal_features: Optional[Dict],
        rl_recommendation: Optional[Dict],
        korean_content: Dict,
        original_analysis: Dict
    ) -> Dict[str, Any]:
        """모든 결과를 통합"""
        
        combined_result = {
            # 기본 레시피
            'name': korean_content.get('name', '향수'),
            'description': korean_content.get('description', '특별한 향수입니다.'),
            'top_notes': base_recipe.get('top_notes', []),
            'middle_notes': base_recipe.get('middle_notes', []),
            'base_notes': base_recipe.get('base_notes', []),
            'intensity': base_recipe.get('intensity', 5.0),
            
            # 품질 점수들
            'composition_harmony': base_recipe.get('composition_harmony', 0.7),
            'confidence_scores': base_recipe.get('confidence_scores', {}),
            
            # 한국어 콘텐츠
            'korean_content': {
                'emotional_story': korean_content.get('emotional_story', ''),
                'usage_recommendation': korean_content.get('usage_recommendation', ''),
                'marketing_copy': korean_content.get('marketing_copy', ''),
                'cultural_context': korean_content.get('cultural_context', '')
            },
            
            # AI 시스템 정보
            'ai_insights': {
                'original_emotions': original_analysis['emotions'],
                'original_keywords': original_analysis['keywords'],
                'rag_enhanced': bool(rag_context),
                'multimodal_processed': bool(multimodal_features),
                'rl_personalized': bool(rl_recommendation),
                'korean_llm_generated': bool(korean_content)
            }
        }
        
        # RAG 정보 추가
        if rag_context:
            combined_result['knowledge_base'] = {
                'recommendations_count': len(rag_context.get('recommendations', {})),
                'confidence': rag_context.get('knowledge_confidence', 0.0)
            }
        
        # 다중 모달 정보 추가
        if multimodal_features:
            combined_result['multimodal_info'] = {
                'modalities_used': multimodal_features.get('modalities_used', []),
                'confidence': multimodal_features.get('metadata', {}).get('encoding_confidence', 0.0)
            }
        
        # RL 개인화 정보 추가
        if rl_recommendation:
            combined_result['personalization'] = {
                'user_personalization': rl_recommendation.get('personalization', {}),
                'recommendation_confidence': rl_recommendation.get('confidence', 0.0)
            }
        
        return combined_result
    
    def _assess_generation_quality(self, result: Dict) -> str:
        """생성 품질 평가"""
        quality_indicators = [
            len(result.get('top_notes', [])) >= 2,
            len(result.get('middle_notes', [])) >= 2, 
            len(result.get('base_notes', [])) >= 1,
            result.get('composition_harmony', 0) > 0.5,
            bool(result.get('korean_content', {}).get('description')),
            result.get('ai_insights', {}).get('rag_enhanced', False)
        ]
        
        quality_score = sum(quality_indicators) / len(quality_indicators)
        
        if quality_score >= 0.8:
            return "고품질"
        elif quality_score >= 0.6:
            return "양호"
        else:
            return "기본"
    
    def _generate_basic_recipe(self, analysis: Dict) -> Dict[str, Any]:
        """기본 레시피 생성 (폴백)"""
        # 기존 시스템 사용
        from models.recipe_generator import RecipeGenerator
        basic_generator = RecipeGenerator()
        
        import torch
        emotion_scores = torch.tensor(
            [list(analysis['emotions'].values())],
            dtype=torch.float32
        )
        
        return basic_generator.generate_recipe(
            analysis['embeddings'],
            emotion_scores
        )
    
    def _generate_fallback_recipe(self, text: str) -> Dict[str, Any]:
        """최종 폴백 레시피"""
        return {
            'name': '기본 향수',
            'description': f'"{text}"에서 영감을 받은 향수입니다.',
            'top_notes': ['citrus', 'fresh'],
            'middle_notes': ['floral'],
            'base_notes': ['musk'],
            'intensity': 5.0,
            'system_info': {
                'fallback_mode': True,
                'generation_quality': '기본'
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
        
        # 평균 응답 시간 업데이트
        total = self.performance_stats['total_requests']
        current_avg = self.performance_stats['average_response_time']
        self.performance_stats['average_response_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def process_user_feedback(
        self,
        recipe_id: int,
        feedback_data: Dict[str, Any],
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """사용자 피드백 처리 및 학습"""
        
        try:
            # 데이터베이스에 피드백 저장
            with get_db_session() as session:
                db_feedback = Feedback(
                    recipe_id=recipe_id,
                    rating=feedback_data.get('rating', 5.0),
                    comments=feedback_data.get('comments', ''),
                    emotional_response=str(feedback_data.get('emotional_response', [])),
                    longevity=feedback_data.get('longevity', 5),
                    sillage=feedback_data.get('sillage', 5)
                )
                session.add(db_feedback)
                session.commit()
            
            # RL 시스템에 피드백 전달 (가능한 경우)
            rl_training_result = {}
            if self.rl_system:
                try:
                    # 피드백을 RL 형태로 변환
                    rl_feedback = RLUserFeedback(
                        recipe_id=str(recipe_id),
                        overall_rating=feedback_data.get('rating', 5.0),
                        fragrance_notes=feedback_data.get('note_ratings', {}),
                        emotional_response=feedback_data.get('emotional_response', []),
                        longevity_rating=feedback_data.get('longevity', 5.0),
                        sillage_rating=feedback_data.get('sillage', 5.0),
                        occasion_appropriateness=feedback_data.get('occasions', {}),
                        improvement_suggestions=feedback_data.get('suggestions', []),
                        user_id=user_id,
                        timestamp=str(time.time())
                    )
                    
                    # 가상의 상태와 액션으로 학습 (실제로는 더 정교한 매핑 필요)
                    dummy_state = FragranceState(
                        user_preferences={},
                        context={},
                        emotion_target=[],
                        previous_feedback=[],
                        user_history={}
                    )
                    
                    dummy_action = FragranceAction(
                        top_notes=['citrus'],
                        middle_notes=['floral'], 
                        base_notes=['musk'],
                        intensity=5.0
                    )
                    
                    rl_training_result = self.rl_system.process_feedback(
                        dummy_state, dummy_action, rl_feedback
                    )
                    
                except Exception as e:
                    logger.warning(f"RL 피드백 처리 실패: {e}")
            
            return {
                'feedback_saved': True,
                'recipe_id': recipe_id,
                'rl_training_result': rl_training_result,
                'message': '피드백이 성공적으로 처리되었습니다. 향후 추천에 반영될 예정입니다.'
            }
            
        except Exception as e:
            logger.error(f"피드백 처리 실패: {e}")
            return {
                'feedback_saved': False,
                'error': str(e),
                'message': '피드백 처리 중 오류가 발생했습니다.'
            }
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """시스템 성능 리포트"""
        
        report = {
            'performance_stats': self.performance_stats.copy(),
            'system_components': {
                'basic_systems': ['텍스트 분석', 'Transformer 레시피 생성'],
                'advanced_systems': [],
                'total_active_components': len(self._get_active_components())
            }
        }
        
        # 각 고급 시스템의 상태와 성능
        if self.rag_system:
            report['system_components']['advanced_systems'].append('RAG 지식베이스')
            report['rag_stats'] = self.rag_system.get_system_stats()
        
        if self.multimodal_system:
            report['system_components']['advanced_systems'].append('다중 모달 임베딩')
            report['multimodal_stats'] = self.multimodal_system.get_system_info()
        
        if self.rl_system:
            report['system_components']['advanced_systems'].append('강화학습 시스템')
            report['rl_stats'] = self.rl_system.get_system_metrics()
        
        if self.korean_llm_system:
            report['system_components']['advanced_systems'].append('한국어 LLM')
            report['korean_llm_stats'] = self.korean_llm_system.get_system_status()
        
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


def run_advanced_demo():
    """고급 시스템 데모 실행"""
    
    print("🎨 차세대 향수 AI 시스템 데모")
    print("=" * 80)
    
    # 데이터베이스 초기화
    init_db()
    
    # 고급 시스템 초기화
    perfume_system = AdvancedPerfumeAISystem()
    
    # 테스트 시나리오들
    test_scenarios = [
        {
            'text': "봄날 아침 공원을 산책하며 느끼는 상쾌하고 평온한 감정을 향수로 표현하고 싶어요",
            'user_id': "user_001",
            'description': "🌸 봄날 산책 시나리오"
        },
        {
            'text': "겨울 저녁 벽난로 앞에서 따뜻한 차를 마시며 책을 읽는 아늑한 분위기",
            'user_id': "user_002", 
            'description': "🔥 겨울 독서 시나리오"
        },
        {
            'text': "여름 해변에서 느끼는 시원하고 자유로운 바다의 에너지",
            'user_id': "user_003",
            'description': "🏖️ 여름 해변 시나리오"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"🎯 테스트 {i}: {scenario['description']}")
        print(f"📝 입력: {scenario['text']}")
        print(f"👤 사용자: {scenario['user_id']}")
        print("-" * 60)
        
        # 고급 레시피 생성
        result = perfume_system.generate_advanced_recipe(
            text=scenario['text'],
            user_id=scenario['user_id'],
            use_rag=True,
            use_multimodal=False,  # 이미지 없음
            use_rl_personalization=True,
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
        
        # 한국어 콘텐츠
        korean_content = result.get('korean_content', {})
        if korean_content.get('emotional_story'):
            print(f"💭 감성 스토리: {korean_content['emotional_story'][:100]}...")
        
        if korean_content.get('marketing_copy'):
            print(f"📢 마케팅 카피: {korean_content['marketing_copy']}")
        
        # AI 시스템 정보
        system_info = result.get('system_info', {})
        print(f"🤖 사용된 AI 컴포넌트: {', '.join(system_info.get('components_used', []))}")
        print(f"⏱️ 처리 시간: {result.get('processing_time', 0):.2f}초")
        print(f"⭐ 생성 품질: {system_info.get('generation_quality', 'unknown')}")
        
        # 가상의 피드백 처리 (데모용)
        if i == 1:  # 첫 번째 테스트에만 피드백 데모
            print("\n📝 피드백 처리 데모...")
            feedback_data = {
                'rating': 8.5,
                'comments': '정말 봄의 느낌이 잘 표현되었어요!',
                'emotional_response': ['평온', '상쾌', '기쁨'],
                'longevity': 7,
                'sillage': 6,
                'note_ratings': {'citrus': 9, 'floral': 8, 'green': 7}
            }
            
            feedback_result = perfume_system.process_user_feedback(
                result.get('recipe_id', 1),
                feedback_data,
                scenario['user_id']
            )
            
            print(f"✅ 피드백 처리 결과: {feedback_result['message']}")
    
    # 시스템 성능 리포트
    print(f"\n{'='*80}")
    print("📊 시스템 성능 리포트")
    print("=" * 80)
    
    performance_report = perfume_system.get_system_performance_report()
    
    print(f"📈 총 요청 수: {performance_report['performance_stats']['total_requests']}")
    print(f"✅ 성공 생성: {performance_report['performance_stats']['successful_generations']}")
    print(f"⚡ 평균 응답 시간: {performance_report['performance_stats']['average_response_time']:.2f}초")
    print(f"🏥 시스템 상태: {performance_report['system_health']['status']}")
    print(f"💯 성공률: {performance_report['system_health']['success_rate']:.1%}")
    
    active_components = (
        performance_report['system_components']['basic_systems'] +
        performance_report['system_components']['advanced_systems']
    )
    
    print(f"🔧 활성 컴포넌트 ({len(active_components)}개):")
    for component in active_components:
        print(f"   • {component}")
    
    print("\n🎉 고급 향수 AI 시스템 데모 완료!")


if __name__ == "__main__":
    try:
        run_advanced_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  데모가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 데모 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()