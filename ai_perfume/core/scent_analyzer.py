from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ScentProfile:
    """향 프로파일 데이터 클래스"""
    category: str
    score: float


@dataclass
class IntensityInfo:
    """강도 정보 데이터 클래스"""
    level: int
    description: str


@dataclass
class EnvironmentInfo:
    """환경 정보 데이터 클래스"""
    time_of_day: Optional[str] = None
    season: Optional[str] = None
    weather: Optional[str] = None
    location: Optional[str] = None


@dataclass
class ScentAnalysis:
    """향 분석 결과 데이터 클래스"""
    scent_profile: List[Tuple[str, float]]
    intensity: Dict[str, int | str]
    texture: List[str]
    environment: Dict[str, Optional[str]]


class ScentAnalyzer:
    """향 분석 클래스"""
    
    def __init__(self) -> None:
        self.scent_keywords: Dict[str, List[str]] = {
            "woody": ["나무", "우디", "삼나무", "편백", "샌달우드", "건조한", "묵직한"],
            "floral": ["꽃", "플로럴", "장미", "자스민", "라일락", "달콤한"],
            "citrus": ["시트러스", "레몬", "라임", "오렌지", "상큼한", "신선한"],
            "green": ["풀", "이끼", "자연", "청량한", "신선한", "허브"],
            "marine": ["바다", "해변", "소금", "시원한", "청량한"],
            "spicy": ["스파이시", "후추", "계피", "따뜻한", "자극적인"],
            "leather": ["가죽", "스모키", "건조한", "무거운"],
            "powder": ["파우더리", "부드러운", "포근한", "달콤한"],
            "musk": ["머스크", "포근한", "달콤한", "묵직한"],
            "vanilla": ["바닐라", "달콤한", "부드러운", "따뜻한"]
        }
        
        self.intensity_keywords: Dict[str, List[str]] = {
            "strong": ["강한", "강렬한", "자극적인", "진한"],
            "moderate": ["적당한", "중간", "균형잡힌"],
            "weak": ["약한", "은은한", "섬세한", "연한"]
        }
        
        self.texture_keywords: Dict[str, List[str]] = {
            "dry": ["건조한", "드라이한", "메마른"],
            "fresh": ["신선한", "프레시", "상쾌한"],
            "heavy": ["무거운", "묵직한", "진한"],
            "light": ["가벼운", "라이트한", "산뜻한"]
        }
    
    def analyze_description(self, description: str) -> Dict[str, any]:
        """향 설명을 분석하여 구성 요소 추출"""
        description = description.lower()
        
        # 주요 향 특성 분석
        scent_profile = self._analyze_scent_profile(description)
        
        # 강도 분석
        intensity = self._analyze_intensity(description)
        
        # 질감 분석
        texture = self._analyze_texture(description)
        
        # 시간대/환경 분석
        environment = self._analyze_environment(description)
        
        return {
            'scent_profile': scent_profile,
            'intensity': intensity,
            'texture': texture,
            'environment': environment
        }
    
    def _analyze_scent_profile(self, description: str) -> List[Tuple[str, float]]:
        """주요 향 프로파일 분석"""
        profile = []
        
        for category, keywords in self.scent_keywords.items():
            score = 0
            matches = 0
            for keyword in keywords:
                if keyword in description:
                    matches += 1
                    # 키워드가 여러 번 등장하면 가중치 증가
                    score += description.count(keyword) * 0.5
            
            if matches > 0:
                # 정규화된 점수 계산 (0-1 범위)
                normalized_score = min(score / len(keywords), 1.0)
                profile.append((category, normalized_score))
        
        # 점수 기준 내림차순 정렬
        profile.sort(key=lambda x: x[1], reverse=True)
        return profile
    
    def _analyze_intensity(self, description: str) -> Dict[str, int | str]:
        """향의 강도 분석"""
        intensity_scores: Dict[str, int | str] = {
            'level': 5,  # 기본값
            'description': 'moderate'
        }
        
        for level, keywords in self.intensity_keywords.items():
            for keyword in keywords:
                if keyword in description:
                    if level == "strong":
                        intensity_scores['level'] = 8
                        intensity_scores['description'] = 'strong'
                    elif level == "weak":
                        intensity_scores['level'] = 3
                        intensity_scores['description'] = 'weak'
                    else:
                        intensity_scores['level'] = 5
                        intensity_scores['description'] = 'moderate'
                    return intensity_scores
                    
        return intensity_scores
    
    def _analyze_texture(self, description: str) -> List[str]:
        """향의 질감 특성 분석"""
        textures = []
        
        for texture, keywords in self.texture_keywords.items():
            for keyword in keywords:
                if keyword in description and texture not in textures:
                    textures.append(texture)
                    
        return textures
    
    def _analyze_environment(self, description: str) -> Dict[str, Optional[str]]:
        """환경적 맥락 분석"""
        environment: Dict[str, Optional[str]] = {
            'time_of_day': None,
            'season': None,
            'weather': None,
            'location': None
        }
        
        # 시간대 분석
        time_patterns = {
            'morning': ['아침', '새벽'],
            'afternoon': ['오후', '낮'],
            'evening': ['저녁', '해질녘'],
            'night': ['밤', '야간']
        }
        
        # 계절 분석
        season_patterns = {
            'spring': ['봄', '춘'],
            'summer': ['여름', '하'],
            'autumn': ['가을', '추'],
            'winter': ['겨울', '동']
        }
        
        # 날씨 분석
        weather_patterns = {
            'rainy': ['비', '우천'],
            'sunny': ['맑은', '햇살'],
            'cloudy': ['흐린', '구름'],
            'snowy': ['눈', '설']
        }
        
        # 장소 분석
        location_patterns = {
            'indoor': ['실내', '방', '집'],
            'outdoor': ['야외', '거리', '공원'],
            'nature': ['숲', '바다', '산'],
            'urban': ['도시', '거리', '빌딩']
        }
        
        # 각 패턴 검사
        for category, patterns in [
            ('time_of_day', time_patterns),
            ('season', season_patterns),
            ('weather', weather_patterns),
            ('location', location_patterns)
        ]:
            for key, keywords in patterns.items():
                if any(keyword in description for keyword in keywords):
                    environment[category] = key
                    break
                    
        return environment
    
    def get_material_recommendations(self, analysis: Dict[str, any]) -> Dict[str, any]:
        """분석 결과를 바탕으로 원료 추천"""
        recommendations = {
            'primary': [],    # 주요 원료
            'secondary': [],  # 보조 원료
            'fixatives': []   # 고정제
        }
        
        # 주요 향 프로파일에 따른 원료 선정
        scent_material_mapping = {
            'woody': {
                'primary': ['삼나무', '샌달우드', '파촐리'],
                'secondary': ['베티버', '시더우드'],
                'fixatives': ['벤조인', '바닐라']
            },
            'floral': {
                'primary': ['장미', '자스민', '일랑일랑'],
                'secondary': ['라벤더', '제라늄'],
                'fixatives': ['머스크', '앰브레트']
            },
            'citrus': {
                'primary': ['베르가못', '레몬', '오렌지'],
                'secondary': ['그레이프프룻', '만다린'],
                'fixatives': ['페티그레인', '네롤리']
            }
            # ... 다른 카테고리들도 유사하게 정의
        }
        
        # 상위 2개의 향 프로파일에 대해 원료 선정
        for category, score in analysis['scent_profile'][:2]:
            if category in scent_material_mapping:
                materials = scent_material_mapping[category]
                recommendations['primary'].extend(materials['primary'])
                recommendations['secondary'].extend(materials['secondary'])
                recommendations['fixatives'].extend(materials['fixatives'])
        
        # 강도에 따른 원료 비율 조정
        intensity = analysis['intensity']['level']
        if intensity > 7:
            recommendations['concentration_factor'] = 1.2
        elif intensity < 4:
            recommendations['concentration_factor'] = 0.8
        else:
            recommendations['concentration_factor'] = 1.0
            
        return recommendations 