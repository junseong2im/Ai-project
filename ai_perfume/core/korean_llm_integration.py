from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json
import asyncio
import logging

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        GPT2LMHeadModel, PreTrainedTokenizerFast,
        AutoModelForSequenceClassification,
        pipeline
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers library not available.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not available.")


logger = logging.getLogger(__name__)


@dataclass
class FragranceDescription:
    """향수 설명 구조"""
    name: str
    description: str
    emotional_story: str
    usage_recommendation: str
    cultural_context: str
    marketing_copy: str


@dataclass
class LLMResponse:
    """LLM 응답 구조"""
    content: str
    model_name: str
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]


class KoreanLLMManager:
    """한국어 특화 LLM 관리 시스템"""
    
    def __init__(self):
        self.available_models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize_models()
    
    def initialize_models(self):
        """사용 가능한 한국어 LLM 모델들 초기화"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Using fallback text generation.")
            return
        
        # KoGPT 계열 모델들
        self.kogpt_models = {
            'kogpt2-base': {
                'model_name': 'skt/kogpt2-base-v2',
                'description': '한국어 기본 생성 모델',
                'max_length': 1024,
                'use_case': ['general_text', 'creative_writing']
            },
            'kobart-base': {
                'model_name': 'gogamza/kobart-base-v2',
                'description': '한국어 요약 및 생성 모델',
                'max_length': 512,
                'use_case': ['summarization', 'text_generation']
            }
        }
        
        # Solar 계열 (Upstage)
        self.solar_models = {
            'solar-1-mini': {
                'model_name': 'upstage/SOLAR-1-mini-chat',
                'description': 'Upstage의 한국어 특화 모델',
                'max_length': 2048,
                'use_case': ['conversation', 'qa', 'creative_writing']
            }
        }
        
        # HyperClova X (Naver) - API 기반
        self.hyperclova_config = {
            'model_name': 'HyperClova X',
            'description': 'Naver의 초거대 한국어 AI',
            'api_endpoint': None,  # 실제 사용시 설정 필요
            'use_case': ['enterprise', 'advanced_reasoning']
        }
        
        # 실제 모델 로딩 시도
        self._load_available_models()
    
    def _load_available_models(self):
        """사용 가능한 모델들 실제 로딩"""
        loaded_models = {}
        
        # KoGPT2 로딩 시도
        try:
            kogpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
                'skt/kogpt2-base-v2',
                bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                pad_token='<pad>', mask_token='<mask>'
            )
            kogpt_model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
            kogpt_model.to(self.device)
            
            loaded_models['kogpt2'] = {
                'model': kogpt_model,
                'tokenizer': kogpt_tokenizer,
                'config': self.kogpt_models['kogpt2-base']
            }
            logger.info("KoGPT2 모델 로딩 완료")
            
        except Exception as e:
            logger.warning(f"KoGPT2 로딩 실패: {e}")
        
        # KoBART 로딩 시도  
        try:
            kobart_tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v2')
            kobart_model = AutoModelForCausalLM.from_pretrained('gogamza/kobart-base-v2')
            kobart_model.to(self.device)
            
            loaded_models['kobart'] = {
                'model': kobart_model,
                'tokenizer': kobart_tokenizer,
                'config': self.kogpt_models['kobart-base']
            }
            logger.info("KoBART 모델 로딩 완료")
            
        except Exception as e:
            logger.warning(f"KoBART 로딩 실패: {e}")
        
        self.available_models = loaded_models
    
    def generate_text(
        self, 
        prompt: str, 
        model_name: str = 'kogpt2',
        max_length: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> LLMResponse:
        """텍스트 생성"""
        
        if model_name not in self.available_models:
            return self._fallback_text_generation(prompt)
        
        import time
        start_time = time.time()
        
        try:
            model_info = self.available_models[model_name]
            model = model_info['model']
            tokenizer = model_info['tokenizer']
            
            # 입력 토큰화
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # 생성 파라미터
            generation_config = {
                'max_length': min(max_length, input_ids.shape[1] + 100),
                'temperature': temperature,
                'top_p': top_p,
                'repetition_penalty': repetition_penalty,
                'do_sample': True,
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id
            }
            
            # 텍스트 생성
            with torch.no_grad():
                outputs = model.generate(input_ids, **generation_config)
            
            # 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 부분 제거
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            processing_time = time.time() - start_time
            
            return LLMResponse(
                content=generated_text,
                model_name=model_name,
                confidence_score=0.8,  # 간단한 휴리스틱
                processing_time=processing_time,
                metadata={
                    'prompt_length': len(prompt),
                    'generated_length': len(generated_text),
                    'temperature': temperature,
                    'top_p': top_p
                }
            )
            
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류: {e}")
            return self._fallback_text_generation(prompt)
    
    def _fallback_text_generation(self, prompt: str) -> LLMResponse:
        """대체 텍스트 생성 (모델 로딩 실패시)"""
        # 간단한 템플릿 기반 생성
        fallback_responses = {
            '향수': '이 향수는 독특하고 매혹적인 향을 지니고 있습니다. 상쾌한 시작과 깊이 있는 마무리가 어우러져 특별한 경험을 선사합니다.',
            '감정': '이 조합은 마음을 편안하게 하고 긍정적인 에너지를 불러일으킵니다.',
            '계절': '계절의 특성을 잘 반영한 조화로운 향수입니다.',
            '추천': '일상에서 특별함을 느끼고 싶을 때 추천하는 향수입니다.'
        }
        
        # 키워드 기반 응답 선택
        response = fallback_responses['향수']  # 기본값
        for keyword, template in fallback_responses.items():
            if keyword in prompt:
                response = template
                break
        
        return LLMResponse(
            content=response,
            model_name='fallback',
            confidence_score=0.3,
            processing_time=0.001,
            metadata={'method': 'template_based'}
        )


class FragranceDescriptionGenerator:
    """향수 설명 생성 전문 시스템"""
    
    def __init__(self, llm_manager: KoreanLLMManager):
        self.llm_manager = llm_manager
        self.description_templates = self._load_description_templates()
    
    def _load_description_templates(self) -> Dict[str, str]:
        """설명 생성 템플릿 로드"""
        return {
            'name_generation': """
다음 향수의 특성을 바탕으로 매혹적이고 기억하기 쉬운 한국어 향수 이름을 제안해주세요:

향료 구성: {notes}
감정/분위기: {emotions}  
사용 상황: {context}

한국의 감성과 문화를 반영한 시적이고 우아한 이름을 3개 제안해주세요:
""",
            
            'description_generation': """
다음 향수에 대한 매혹적이고 감성적인 설명을 작성해주세요:

향수 이름: {name}
주요 향료: {notes}
감정 키워드: {emotions}
타겟 고객: {target_audience}

한국어의 아름다운 표현을 사용하여 이 향수만의 독특함과 매력을 200-300자로 표현해주세요:
""",
            
            'story_generation': """
다음 향수에 얽힌 감성적인 이야기를 창작해주세요:

향수 이름: {name}  
향의 특성: {fragrance_profile}
계절/시간: {season_time}

이 향수를 착용했을 때의 순간이나 기억을 아름다운 산문으로 표현해주세요 (150-200자):
""",
            
            'usage_recommendation': """  
다음 향수의 사용법과 어울리는 상황을 추천해주세요:

향수명: {name}
향료 특성: {characteristics}
지속성: {longevity}
확산력: {sillage}

언제, 어디서, 어떤 상황에 이 향수를 사용하면 좋을지 구체적이고 실용적인 조언을 해주세요:
""",
            
            'cultural_context': """
다음 향수를 한국의 문화적 맥락에서 해석하고 설명해주세요:

향수 구성: {composition}
감성 키워드: {emotions}
계절감: {seasonality}

한국의 전통, 자연, 현대적 감성과 어떻게 연결될 수 있는지 설명해주세요:
""",
            
            'marketing_copy': """
다음 향수의 매력적인 마케팅 카피를 작성해주세요:

제품명: {name}
핵심 메시지: {key_message}
타겟 고객: {target}
브랜드 톤: {brand_tone}

감정을 자극하고 구매 욕구를 불러일으키는 짧고 임팩트 있는 카피를 만들어주세요:
"""
        }
    
    def generate_complete_description(
        self,
        fragrance_data: Dict[str, Any],
        model_preference: str = 'kogpt2'
    ) -> FragranceDescription:
        """완전한 향수 설명 생성"""
        
        # 1. 향수 이름 생성
        name_prompt = self.description_templates['name_generation'].format(
            notes=', '.join(fragrance_data.get('all_notes', [])),
            emotions=', '.join(fragrance_data.get('emotions', [])),
            context=fragrance_data.get('context', '일상')
        )
        
        name_response = self.llm_manager.generate_text(
            name_prompt, model_preference, max_length=100
        )
        
        # 생성된 이름 중 첫 번째 선택 (간단화)
        suggested_names = name_response.content.split('\n')
        perfume_name = suggested_names[0].strip() if suggested_names else "향수"
        
        # 숫자나 특수문자 제거
        import re
        perfume_name = re.sub(r'[0-9.\-]', '', perfume_name).strip()
        
        # 2. 기본 설명 생성
        desc_prompt = self.description_templates['description_generation'].format(
            name=perfume_name,
            notes=', '.join(fragrance_data.get('all_notes', [])),
            emotions=', '.join(fragrance_data.get('emotions', [])),
            target_audience=fragrance_data.get('target_audience', '모든 연령')
        )
        
        description_response = self.llm_manager.generate_text(
            desc_prompt, model_preference, max_length=150
        )
        
        # 3. 감성적 스토리 생성
        story_prompt = self.description_templates['story_generation'].format(
            name=perfume_name,
            fragrance_profile=fragrance_data.get('fragrance_profile', '조화로운 향'),
            season_time=fragrance_data.get('season_time', '봄날 오후')
        )
        
        story_response = self.llm_manager.generate_text(
            story_prompt, model_preference, max_length=120
        )
        
        # 4. 사용법 추천 생성
        usage_prompt = self.description_templates['usage_recommendation'].format(
            name=perfume_name,
            characteristics=fragrance_data.get('characteristics', '우아하고 세련된'),
            longevity=fragrance_data.get('longevity', 5),
            sillage=fragrance_data.get('sillage', 5)
        )
        
        usage_response = self.llm_manager.generate_text(
            usage_prompt, model_preference, max_length=100
        )
        
        # 5. 문화적 맥락 생성
        cultural_prompt = self.description_templates['cultural_context'].format(
            composition=', '.join(fragrance_data.get('all_notes', [])),
            emotions=', '.join(fragrance_data.get('emotions', [])),
            seasonality=fragrance_data.get('seasonality', '사계절')
        )
        
        cultural_response = self.llm_manager.generate_text(
            cultural_prompt, model_preference, max_length=120
        )
        
        # 6. 마케팅 카피 생성
        marketing_prompt = self.description_templates['marketing_copy'].format(
            name=perfume_name,
            key_message=fragrance_data.get('key_message', '당신만의 특별한 향'),
            target=fragrance_data.get('target_audience', '현대인'),
            brand_tone=fragrance_data.get('brand_tone', '세련되고 감성적인')
        )
        
        marketing_response = self.llm_manager.generate_text(
            marketing_prompt, model_preference, max_length=80
        )
        
        return FragranceDescription(
            name=perfume_name,
            description=description_response.content.strip(),
            emotional_story=story_response.content.strip(),
            usage_recommendation=usage_response.content.strip(),
            cultural_context=cultural_response.content.strip(),
            marketing_copy=marketing_response.content.strip()
        )
    
    def generate_single_aspect(
        self,
        aspect: str,
        fragrance_data: Dict[str, Any],
        model_preference: str = 'kogpt2'
    ) -> str:
        """특정 측면의 설명만 생성"""
        
        if aspect not in self.description_templates:
            return f"{aspect}에 대한 템플릿을 찾을 수 없습니다."
        
        # 템플릿에 데이터 적용
        try:
            prompt = self.description_templates[aspect].format(**fragrance_data)
        except KeyError as e:
            return f"필요한 데이터가 부족합니다: {e}"
        
        response = self.llm_manager.generate_text(
            prompt, model_preference, max_length=150
        )
        
        return response.content.strip()


class KoreanEmotionClassifier:
    """한국어 감정 분류기"""
    
    def __init__(self):
        self.emotion_categories = {
            'positive': ['기쁨', '행복', '설렘', '평온', '만족', '희망'],
            'calm': ['차분', '안정', '평화', '고요', '여유', '편안'],
            'energetic': ['활기', '에너지', '역동', '상쾌', '생동감', '활력'],
            'romantic': ['로맨틱', '사랑', '낭만', '설렘', '달콤', '감성'],
            'sophisticated': ['세련', '우아', '고급', '품격', '성숙', '지적'],
            'mysterious': ['신비', '몽환', '미스터리', '깊이', '복합', '은밀'],
            'fresh': ['신선', '깨끗', '상쾌', '순수', '맑음', '청량'],
            'warm': ['따뜻', '포근', '아늑', '부드러움', '온화', '정겨움']
        }
        
        self.korean_emotion_words = {}
        for category, words in self.emotion_categories.items():
            for word in words:
                self.korean_emotion_words[word] = category
    
    def classify_emotions(self, text: str) -> Dict[str, float]:
        """텍스트에서 감정 분류"""
        emotion_scores = {category: 0.0 for category in self.emotion_categories.keys()}
        
        # 간단한 키워드 기반 분류
        words = text.split()
        
        for word in words:
            for emotion_word, category in self.korean_emotion_words.items():
                if emotion_word in word or word in emotion_word:
                    emotion_scores[category] += 1.0
        
        # 정규화
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
        else:
            # 기본값 설정
            emotion_scores['positive'] = 0.5
            emotion_scores['fresh'] = 0.3
            emotion_scores['calm'] = 0.2
        
        return emotion_scores
    
    def get_dominant_emotions(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """주요 감정들 추출"""
        emotion_scores = self.classify_emotions(text)
        
        sorted_emotions = sorted(
            emotion_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_emotions[:top_k]


class KoreanFragranceLLMSystem:
    """한국어 특화 향수 LLM 통합 시스템"""
    
    def __init__(self):
        self.llm_manager = KoreanLLMManager()
        self.description_generator = FragranceDescriptionGenerator(self.llm_manager)
        self.emotion_classifier = KoreanEmotionClassifier()
        
        logger.info("Korean Fragrance LLM System 초기화 완료")
    
    def generate_korean_fragrance_content(
        self,
        recipe_data: Dict[str, Any],
        content_types: List[str] = None,
        model_preference: str = 'kogpt2'
    ) -> Dict[str, Any]:
        """한국어 향수 콘텐츠 종합 생성"""
        
        if content_types is None:
            content_types = ['name', 'description', 'story', 'usage', 'marketing']
        
        # 입력 데이터 전처리
        processed_data = self._preprocess_recipe_data(recipe_data)
        
        results = {}
        
        # 완전한 설명 생성
        if 'all' in content_types or len(content_types) >= 4:
            full_description = self.description_generator.generate_complete_description(
                processed_data, model_preference
            )
            
            results.update({
                'name': full_description.name,
                'description': full_description.description,
                'emotional_story': full_description.emotional_story,
                'usage_recommendation': full_description.usage_recommendation,
                'cultural_context': full_description.cultural_context,
                'marketing_copy': full_description.marketing_copy
            })
        
        else:
            # 개별 콘텐츠 생성
            content_mapping = {
                'name': 'name_generation',
                'description': 'description_generation',
                'story': 'story_generation', 
                'usage': 'usage_recommendation',
                'cultural': 'cultural_context',
                'marketing': 'marketing_copy'
            }
            
            for content_type in content_types:
                if content_type in content_mapping:
                    aspect = content_mapping[content_type]
                    result = self.description_generator.generate_single_aspect(
                        aspect, processed_data, model_preference
                    )
                    results[content_type] = result
        
        # 감정 분석 추가
        if 'description' in results:
            emotion_analysis = self.emotion_classifier.classify_emotions(
                results['description']
            )
            results['emotion_analysis'] = emotion_analysis
            results['dominant_emotions'] = self.emotion_classifier.get_dominant_emotions(
                results['description']
            )
        
        # 메타데이터 추가
        results['metadata'] = {
            'model_used': model_preference,
            'generation_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown',
            'content_types_generated': content_types,
            'korean_processing': True,
            'available_models': list(self.llm_manager.available_models.keys())
        }
        
        return results
    
    def _preprocess_recipe_data(self, recipe_data: Dict[str, Any]) -> Dict[str, Any]:
        """레시피 데이터 전처리"""
        processed = recipe_data.copy()
        
        # 모든 노트 통합
        all_notes = []
        if 'top_notes' in recipe_data:
            all_notes.extend(recipe_data['top_notes'])
        if 'middle_notes' in recipe_data:
            all_notes.extend(recipe_data['middle_notes'])
        if 'base_notes' in recipe_data:
            all_notes.extend(recipe_data['base_notes'])
        
        processed['all_notes'] = all_notes
        
        # 기본값 설정
        defaults = {
            'emotions': ['긍정', '신선'],
            'context': '일상',
            'target_audience': '현대인',
            'fragrance_profile': '조화로운 향',
            'season_time': '봄날 오후',
            'characteristics': '우아하고 세련된',
            'longevity': 5,
            'sillage': 5,
            'seasonality': '사계절',
            'key_message': '당신만의 특별한 향',
            'brand_tone': '세련되고 감성적인'
        }
        
        for key, default_value in defaults.items():
            if key not in processed:
                processed[key] = default_value
        
        return processed
    
    def analyze_user_input_korean(self, user_text: str) -> Dict[str, Any]:
        """한국어 사용자 입력 분석"""
        
        analysis = {
            'original_text': user_text,
            'emotion_classification': self.emotion_classifier.classify_emotions(user_text),
            'dominant_emotions': self.emotion_classifier.get_dominant_emotions(user_text),
            'detected_keywords': [],
            'fragrance_preferences': {},
            'cultural_elements': []
        }
        
        # 키워드 추출 (간단한 방식)
        keywords = self._extract_korean_keywords(user_text)
        analysis['detected_keywords'] = keywords
        
        # 향수 선호도 추출
        preferences = self._extract_fragrance_preferences(user_text)
        analysis['fragrance_preferences'] = preferences
        
        # 한국 문화 요소 감지
        cultural_elements = self._detect_cultural_elements(user_text)
        analysis['cultural_elements'] = cultural_elements
        
        return analysis
    
    def _extract_korean_keywords(self, text: str) -> List[str]:
        """한국어 키워드 추출"""
        # 향수 관련 키워드들
        fragrance_keywords = [
            '향수', '향기', '냄새', '향', '로즈', '라벤더', '바닐라', '시트러스',
            '플로럴', '우디', '머스크', '신선', '달콤', '상쾌', '따뜻', '차가운',
            '봄', '여름', '가을', '겨울', '아침', '오후', '저녁', '밤',
            '로맨틱', '세련', '우아', '활기', '평온', '에너지'
        ]
        
        found_keywords = []
        for keyword in fragrance_keywords:
            if keyword in text:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_fragrance_preferences(self, text: str) -> Dict[str, Any]:
        """향수 선호도 추출"""
        preferences = {
            'intensity': 'medium',
            'longevity': 'medium',
            'preferred_families': [],
            'preferred_occasions': [],
            'preferred_seasons': []
        }
        
        # 강도 선호도
        if any(word in text for word in ['강한', '진한', '짙은']):
            preferences['intensity'] = 'high'
        elif any(word in text for word in ['약한', '연한', '가벼운']):
            preferences['intensity'] = 'low'
        
        # 계절 선호도
        seasons = ['봄', '여름', '가을', '겨울']
        for season in seasons:
            if season in text:
                preferences['preferred_seasons'].append(season)
        
        # 향료 계열 선호도
        families = {
            '시트러스': ['레몬', '오렌지', '자몽', '베르가못'],
            '플로럴': ['장미', '자스민', '라벤더', '꽃'],
            '우디': ['나무', '삼나무', '샌달우드'],
            '머스크': ['머스크', '동물성'],
            '바닐라': ['바닐라', '달콤한']
        }
        
        for family, keywords in families.items():
            if any(keyword in text for keyword in keywords):
                preferences['preferred_families'].append(family)
        
        return preferences
    
    def _detect_cultural_elements(self, text: str) -> List[str]:
        """한국 문화 요소 감지"""
        cultural_keywords = {
            '전통': ['한복', '전통', '고궁', '한옥', '차', '명절'],
            '자연': ['산', '바다', '강', '숲', '꽃', '나무'],
            '계절': ['벚꽃', '단풍', '눈', '매화'],
            '감성': ['정', '한', '그리움', '추억', '향수'],
            '현대': ['도시', '카페', '사무실', '데이트']
        }
        
        detected_elements = []
        for category, keywords in cultural_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected_elements.append(category)
        
        return detected_elements
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보"""
        return {
            'available_models': list(self.llm_manager.available_models.keys()),
            'total_models_configured': len(self.llm_manager.kogpt_models) + len(self.llm_manager.solar_models),
            'successfully_loaded': len(self.llm_manager.available_models),
            'emotion_categories': list(self.emotion_classifier.emotion_categories.keys()),
            'description_templates': list(self.description_generator.description_templates.keys()),
            'device': str(self.llm_manager.device),
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'openai_available': OPENAI_AVAILABLE
        }