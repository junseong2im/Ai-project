#!/usr/bin/env python3
"""
감정 키워드 매핑 공통 유틸리티
여러 모듈에서 중복 사용되는 감정 키워드들을 통합 관리
"""

from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class EmotionKeywordMapper:
    """감정 키워드 매핑 클래스"""
    
    def __init__(self):
        # 한국어-영어 감정 키워드 매핑
        self.emotion_keywords_kr = {
            "love": ["사랑", "로맨틱", "키스", "포옹", "데이트", "만남", "연인", "애정"],
            "sad": ["슬픈", "이별", "눈물", "우울", "외로운", "그리움", "상실", "비극"],
            "fear": ["무서운", "공포", "어둠", "귀신", "죽음", "피", "스릴러", "놀람"],
            "anger": ["화난", "분노", "싸움", "복수", "증오", "적대", "격노"],
            "joy": ["행복", "웃음", "기쁜", "축하", "파티", "즐거운", "희열", "환희"],
            "calm": ["평화", "고요", "차분", "명상", "휴식", "여유", "안정"],
            "tension": ["긴장", "추격", "서스펜스", "스릴", "압박", "조급"],
            "nostalgia": ["그리움", "옛날", "추억", "향수", "회상", "과거"],
            "passion": ["열정", "뜨거운", "격렬한", "강렬한", "에너지", "활력"],
            "mystery": ["신비", "비밀", "수상한", "의문", "미스터리", "은밀"]
        }
        
        # 영어 감정 키워드
        self.emotion_keywords_en = {
            "love": ["romantic", "kiss", "embrace", "date", "lover", "affection", "romance"],
            "sad": ["sad", "melancholy", "tear", "lonely", "sorrow", "grief", "tragic"],
            "fear": ["scary", "horror", "dark", "ghost", "death", "blood", "thriller", "frightening"],
            "anger": ["angry", "rage", "fight", "revenge", "hatred", "fury", "hostile"],
            "joy": ["happy", "laugh", "celebration", "party", "cheerful", "joyful", "delighted"],
            "calm": ["peaceful", "quiet", "serene", "meditation", "rest", "tranquil", "stable"],
            "tension": ["tense", "chase", "suspense", "thrill", "pressure", "urgent"],
            "nostalgia": ["nostalgic", "memory", "past", "reminiscence", "vintage", "old"],
            "passion": ["passionate", "hot", "intense", "energetic", "vigorous", "dynamic"],
            "mystery": ["mysterious", "secret", "suspicious", "enigma", "hidden", "covert"]
        }
        
        # 감정별 색상 매핑
        self.emotion_colors = {
            "love": ["red", "pink", "rose"],
            "sad": ["blue", "gray", "dark"],
            "fear": ["black", "dark", "shadow"],
            "anger": ["red", "orange", "fire"],
            "joy": ["yellow", "bright", "gold"],
            "calm": ["blue", "green", "white"],
            "tension": ["red", "orange", "sharp"],
            "nostalgia": ["sepia", "brown", "vintage"],
            "passion": ["red", "orange", "flame"],
            "mystery": ["purple", "dark", "violet"]
        }
        
        # 감정별 향료 노트 매핑
        self.emotion_scent_notes = {
            "love": {
                "top": ["rose", "jasmine", "neroli"],
                "middle": ["ylang_ylang", "tuberose", "pink_pepper"],
                "base": ["musk", "amber", "vanilla"]
            },
            "sad": {
                "top": ["bergamot", "lavender", "mint"],
                "middle": ["iris", "violet", "lily"],
                "base": ["sandalwood", "cedar", "white_musk"]
            },
            "fear": {
                "top": ["black_pepper", "ginger", "cardamom"],
                "middle": ["incense", "smoke", "dark_woods"],
                "base": ["leather", "tar", "dark_musk"]
            },
            "anger": {
                "top": ["red_pepper", "ginger", "cinnamon"],
                "middle": ["clove", "nutmeg", "hot_spices"],
                "base": ["leather", "tobacco", "fire_smoke"]
            },
            "joy": {
                "top": ["lemon", "orange", "grapefruit"],
                "middle": ["peach", "apple", "green_leaves"],
                "base": ["light_woods", "clean_musk", "honey"]
            },
            "calm": {
                "top": ["eucalyptus", "mint", "tea"],
                "middle": ["chamomile", "lavender", "green"],
                "base": ["sandalwood", "white_woods", "soft_musk"]
            },
            "tension": {
                "top": ["sharp_citrus", "metallic", "ozone"],
                "middle": ["electric_herbs", "sharp_woods"],
                "base": ["synthetic_musk", "industrial", "concrete"]
            },
            "nostalgia": {
                "top": ["vintage_rose", "old_books", "dust"],
                "middle": ["dried_flowers", "old_wood", "memories"],
                "base": ["old_leather", "antique_musk", "time"]
            },
            "passion": {
                "top": ["fire_spices", "hot_pepper", "flame"],
                "middle": ["burning_wood", "hot_leather", "sweat"],
                "base": ["animal_musk", "raw_amber", "heat"]
            },
            "mystery": {
                "top": ["dark_bergamot", "shadow_herbs"],
                "middle": ["night_blooming_jasmine", "dark_rose"],
                "base": ["oud", "dark_amber", "shadow_musk"]
            }
        }
    
    def detect_emotions(self, text: str, language: str = "auto") -> Dict[str, float]:
        """텍스트에서 감정 탐지 및 점수 반환"""
        text_lower = text.lower()
        emotion_scores = {}
        
        # 언어별 키워드 선택
        if language == "auto":
            # 한글 문자가 있으면 한국어, 없으면 영어
            has_korean = any('\uac00' <= char <= '\ud7a3' for char in text)
            keywords = self.emotion_keywords_kr if has_korean else self.emotion_keywords_en
        elif language == "ko":
            keywords = self.emotion_keywords_kr
        else:
            keywords = self.emotion_keywords_en
        
        # 각 감정별 키워드 매칭
        for emotion, emotion_keywords in keywords.items():
            score = sum(1 for keyword in emotion_keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = min(1.0, score / 3)  # 최대 1.0으로 정규화
        
        return emotion_scores
    
    def get_dominant_emotion(self, text: str) -> Tuple[str, float]:
        """가장 강한 감정 반환"""
        emotion_scores = self.detect_emotions(text)
        
        if not emotion_scores:
            return "calm", 0.5  # 기본값
        
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return dominant_emotion
    
    def get_scent_notes_for_emotion(self, emotion: str) -> Dict[str, List[str]]:
        """특정 감정에 맞는 향료 노트 반환"""
        return self.emotion_scent_notes.get(emotion, self.emotion_scent_notes["calm"])
    
    def get_colors_for_emotion(self, emotion: str) -> List[str]:
        """특정 감정에 맞는 색상 반환"""
        return self.emotion_colors.get(emotion, ["neutral"])
    
    def analyze_emotional_profile(self, text: str) -> Dict[str, any]:
        """텍스트의 종합적인 감정 프로필 분석"""
        emotions = self.detect_emotions(text)
        dominant_emotion, confidence = self.get_dominant_emotion(text)
        
        return {
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "all_emotions": emotions,
            "scent_notes": self.get_scent_notes_for_emotion(dominant_emotion),
            "associated_colors": self.get_colors_for_emotion(dominant_emotion),
            "complexity": len(emotions),  # 감정의 복잡도
            "intensity": sum(emotions.values()) / len(emotions) if emotions else 0.5
        }

# 글로벌 인스턴스
emotion_mapper = EmotionKeywordMapper()

def get_emotion_mapper() -> EmotionKeywordMapper:
    """감정 매퍼 인스턴스 반환"""
    return emotion_mapper

# 편의 함수들
def detect_emotions(text: str) -> Dict[str, float]:
    """편의 함수: 감정 탐지"""
    return emotion_mapper.detect_emotions(text)

def get_dominant_emotion(text: str) -> Tuple[str, float]:
    """편의 함수: 주요 감정 반환"""
    return emotion_mapper.get_dominant_emotion(text)

def analyze_emotional_profile(text: str) -> Dict[str, any]:
    """편의 함수: 감정 프로필 분석"""
    return emotion_mapper.analyze_emotional_profile(text)