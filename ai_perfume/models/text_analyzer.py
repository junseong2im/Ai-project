from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import AutoTokenizer, AutoModel

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import Config

class TextAnalyzer:
    """텍스트 분석 클래스"""
    
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        # num_labels 경고를 피하기 위해 명시적으로 설정하지 않음
        self.model = AutoModel.from_pretrained(
            Config.MODEL_NAME,
            ignore_mismatched_sizes=True
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def analyze(self, text: str) -> Dict[str, Any]:
        """텍스트 분석 수행"""
        # 텍스트 토큰화
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=Config.MAX_LENGTH,
            truncation=True,
            padding=True
        )
        
        # BART 모델과의 호환성을 위해 token_type_ids 제거
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        # 입력을 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 텍스트 임베딩 생성
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # 감정 분석
        emotions = self._analyze_emotions(text)
        
        # 키워드 추출
        keywords = self._extract_keywords(text)
        
        return {
            'embeddings': embeddings.cpu(),  # CPU로 다시 이동
            'emotions': emotions,
            'keywords': keywords
        }
    
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """감정 점수 계산"""
        emotions: Dict[str, float] = {}
        text_lower = text.lower()
        
        # 감정 키워드 매핑
        emotion_keywords: Dict[str, List[str]] = {
            'joy': ['행복', '기쁨', '즐거움', '신남', '좋아'],
            'peaceful': ['평화', '고요', '차분', '평온', '조용'],
            'energy': ['활기', '열정', '역동', '힘', '에너지'],
            'melancholy': ['우울', '슬픔', '그리움', '쓸쓸', '멜랑콜리'],
            'mystery': ['신비', '몽환', '환상', '미스터리', '묘한'],
            'romance': ['로맨틱', '사랑', '달콤', '낭만', '설렘'],
            'nature': ['자연', '숲', '바다', '하늘', '꽃'],
            'urban': ['도시', '현대', '세련', '모던', '빌딩']
        }
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotions[emotion] = min(score / len(keywords), 1.0)
            
        return emotions
    
    def _extract_keywords(self, text: str) -> List[str]:
        """간단한 키워드 추출"""
        # 공백으로 분리
        words = text.split()
        # 중복 제거 및 길이 2 이상인 단어만 선택
        keywords = list(set(word for word in words if len(word) >= 2))
        return keywords[:10]  # 상위 10개 키워드만 반환 