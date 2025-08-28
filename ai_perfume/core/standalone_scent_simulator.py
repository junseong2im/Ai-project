#!/usr/bin/env python3
"""
독립형 향 시뮬레이션 시스템 (의존성 없음)
Vercel 배포용으로 최적화된 버전
"""

import random
import json
import time
import math
from typing import Dict, List, Tuple, Optional

class ScentNote:
    """개별 향료 노트"""
    def __init__(self, name: str, category: str, intensity: float, 
                 longevity: float, volatility: str, mood_score: float):
        self.name = name
        self.category = category
        self.intensity = intensity  # 1-10
        self.longevity = longevity  # 1-10 (지속성)
        self.volatility = volatility  # top/middle/base
        self.mood_score = mood_score  # -1(sad) to 1(happy)

class StandaloneScentSimulator:
    """독립형 향 시뮬레이터 (numpy 없이 구동)"""
    
    def __init__(self):
        self.scent_database = self._initialize_scent_database()
        
    def _initialize_scent_database(self) -> Dict[str, List[ScentNote]]:
        """확장된 향료 데이터베이스"""
        database = {
            "top": [
                # 감귤류
                ScentNote("베르가못", "citrus", 8.0, 2.0, "top", 0.7),
                ScentNote("레몬", "citrus", 9.0, 1.5, "top", 0.8),
                ScentNote("라임", "citrus", 8.5, 1.8, "top", 0.9),
                ScentNote("오렌지", "citrus", 7.0, 2.2, "top", 0.8),
                ScentNote("그레이프프룻", "citrus", 8.2, 2.0, "top", 0.6),
                
                # 허브류
                ScentNote("페퍼민트", "herbal", 8.5, 2.5, "top", 0.6),
                ScentNote("라벤더", "herbal", 6.0, 4.0, "top", 0.4),
                ScentNote("로즈마리", "herbal", 7.5, 3.0, "top", 0.3),
                ScentNote("유칼립투스", "herbal", 9.0, 3.0, "top", 0.3),
                ScentNote("바질", "herbal", 7.8, 2.8, "top", 0.2),
                
                # 과일류
                ScentNote("그린 애플", "fruity", 7.0, 2.0, "top", 0.7),
                ScentNote("배", "fruity", 6.5, 2.5, "top", 0.6),
                ScentNote("복숭아", "fruity", 6.8, 2.3, "top", 0.8),
                
                # 해양/수생
                ScentNote("바다 소금", "marine", 6.0, 3.0, "top", 0.2),
                ScentNote("오존", "marine", 7.0, 1.5, "top", 0.5),
                ScentNote("바다 바람", "marine", 6.5, 2.8, "top", 0.4),
                
                # 스파이시
                ScentNote("핑크 페퍼", "spicy", 7.5, 3.0, "top", 0.1),
                ScentNote("진저", "spicy", 8.0, 2.5, "top", 0.3),
                ScentNote("카다몬", "spicy", 7.3, 3.2, "top", 0.2),
            ],
            "middle": [
                # 플로랄
                ScentNote("로즈", "floral", 8.0, 6.0, "middle", 0.8),
                ScentNote("자스민", "floral", 9.0, 7.0, "middle", 0.7),
                ScentNote("릴리", "floral", 6.0, 5.0, "middle", 0.6),
                ScentNote("제라늄", "floral", 7.0, 6.0, "middle", 0.5),
                ScentNote("바이올렛", "floral", 5.5, 4.5, "middle", 0.4),
                ScentNote("프리지아", "floral", 6.5, 5.5, "middle", 0.7),
                ScentNote("피오니", "floral", 6.8, 5.2, "middle", 0.8),
                
                # 스파이시
                ScentNote("시나몬", "spicy", 8.0, 5.0, "middle", 0.2),
                ScentNote("클로브", "spicy", 9.0, 6.0, "middle", -0.1),
                ScentNote("넛맥", "spicy", 7.5, 5.5, "middle", 0.1),
                
                # 구르망
                ScentNote("바닐라", "gourmand", 7.0, 8.0, "middle", 0.6),
                ScentNote("코코넛", "gourmand", 6.0, 5.0, "middle", 0.7),
                ScentNote("카라멜", "gourmand", 7.5, 6.5, "middle", 0.5),
                ScentNote("초콜릿", "gourmand", 8.0, 7.0, "middle", 0.4),
                
                # 그린
                ScentNote("티 리프", "green", 5.0, 4.0, "middle", 0.3),
                ScentNote("대나무", "green", 5.5, 4.5, "middle", 0.4),
                ScentNote("잔디", "green", 5.8, 3.8, "middle", 0.5),
            ],
            "base": [
                # 우디
                ScentNote("샌달우드", "woody", 7.0, 10.0, "base", 0.1),
                ScentNote("시더우드", "woody", 8.0, 9.0, "base", 0.0),
                ScentNote("로즈우드", "woody", 6.5, 8.5, "base", 0.3),
                ScentNote("에보니", "woody", 7.8, 9.5, "base", -0.2),
                
                # 머스크/앰버
                ScentNote("화이트 머스크", "musk", 6.0, 10.0, "base", 0.2),
                ScentNote("블랙 머스크", "musk", 8.5, 10.0, "base", -0.3),
                ScentNote("앰버", "amber", 8.0, 9.5, "base", 0.3),
                ScentNote("아가우드", "amber", 9.5, 10.0, "base", -0.1),
                
                # 어시
                ScentNote("파출리", "earthy", 9.0, 10.0, "base", -0.2),
                ScentNote("오크모스", "earthy", 6.0, 9.0, "base", -0.1),
                ScentNote("베티버", "woody", 7.5, 9.5, "base", -0.3),
                
                # 기타
                ScentNote("통카빈", "gourmand", 8.0, 8.5, "base", 0.4),
                ScentNote("스모키", "smoky", 9.0, 9.0, "base", -0.4),
                ScentNote("레더", "leather", 8.5, 9.2, "base", -0.1),
                ScentNote("인센스", "incense", 7.8, 9.8, "base", -0.2),
            ]
        }
        return database

    def analyze_scene_comprehensive(self, description: str, scene_type: str, 
                                  emotions: List[str] = None) -> Dict[str, float]:
        """향상된 장면 분석"""
        requirements = {
            "intensity": 5.0,
            "mood": 0.0,
            "complexity": 5.0,
            "marine": 0.0,
            "woody": 0.0,
            "floral": 0.0,
            "spicy": 0.0,
            "fresh": 0.0,
            "warm": 0.0,
            "dark": 0.0,
            "gourmand": 0.0,
            "earthy": 0.0,
            "citrus": 0.0
        }
        
        desc = description.lower()
        
        # 시간대 분석
        time_keywords = {
            "새벽": {"fresh": 3, "mood": 0.3},
            "아침": {"citrus": 4, "fresh": 3, "mood": 0.5},
            "점심": {"warm": 2, "intensity": 1},
            "오후": {"warm": 3, "floral": 2},
            "저녁": {"warm": 2, "spicy": 1},
            "밤": {"dark": 4, "intensity": 2, "mood": -0.3},
            "자정": {"dark": 5, "intensity": 3, "mood": -0.5}
        }
        
        for time_word, modifiers in time_keywords.items():
            if time_word in desc:
                for key, value in modifiers.items():
                    requirements[key] += value
        
        # 장소별 세밀한 분석
        location_analysis = {
            # 자연
            "바다": {"marine": 8, "fresh": 6, "mood": 0.4},
            "해변": {"marine": 7, "citrus": 3, "fresh": 5},
            "파도": {"marine": 9, "fresh": 7, "intensity": 3},
            "숲": {"woody": 8, "earthy": 5, "fresh": 4, "mood": 0.2},
            "나무": {"woody": 6, "earthy": 3},
            "꽃밭": {"floral": 9, "fresh": 4, "mood": 0.7},
            "정원": {"floral": 7, "green": 4, "fresh": 3},
            "산": {"woody": 5, "earthy": 4, "fresh": 5},
            "강": {"marine": 5, "fresh": 6},
            
            # 도시
            "도시": {"intensity": 3, "spicy": 2},
            "카페": {"gourmand": 6, "warm": 4, "mood": 0.3},
            "레스토랑": {"gourmand": 5, "spicy": 3, "warm": 3},
            "거리": {"intensity": 2, "dark": 1},
            "옥상": {"fresh": 4, "intensity": 1},
            
            # 실내
            "침실": {"warm": 5, "floral": 3, "mood": 0.4},
            "거실": {"warm": 3, "woody": 2},
            "부엌": {"gourmand": 4, "spicy": 2},
            "욕실": {"fresh": 6, "marine": 3},
            "도서관": {"woody": 4, "earthy": 2, "mood": 0.1}
        }
        
        for location, modifiers in location_analysis.items():
            if location in desc:
                for key, value in modifiers.items():
                    requirements[key] += value
        
        # 날씨 분석
        weather_effects = {
            "비": {"marine": 4, "earthy": 3, "mood": -0.2, "fresh": 2},
            "눈": {"fresh": 5, "marine": 2, "mood": 0.1},
            "바람": {"fresh": 4, "marine": 2},
            "햇살": {"warm": 4, "citrus": 3, "mood": 0.5},
            "구름": {"mood": -0.1, "earthy": 1},
            "안개": {"marine": 3, "mood": -0.2},
            "번개": {"intensity": 5, "dark": 3, "mood": -0.4},
            "천둥": {"intensity": 6, "dark": 4, "mood": -0.5}
        }
        
        for weather, effects in weather_effects.items():
            if weather in desc:
                for key, value in effects.items():
                    requirements[key] += value
        
        # 감정별 정밀 분석
        if emotions:
            emotion_mapping = {
                "love": {"floral": 4, "warm": 3, "gourmand": 2, "mood": 0.6},
                "happy": {"citrus": 3, "floral": 2, "fresh": 2, "mood": 0.5},
                "joy": {"citrus": 4, "fresh": 3, "mood": 0.7},
                "sad": {"earthy": 3, "dark": 3, "mood": -0.5},
                "melancholy": {"woody": 3, "earthy": 2, "mood": -0.3},
                "fear": {"dark": 5, "intensity": 4, "mood": -0.6},
                "anger": {"spicy": 5, "intensity": 5, "dark": 2, "mood": -0.4},
                "tension": {"intensity": 3, "spicy": 2, "dark": 2},
                "calm": {"fresh": 3, "woody": 2, "mood": 0.3},
                "peaceful": {"fresh": 4, "floral": 2, "mood": 0.4}
            }
            
            for emotion in emotions:
                if emotion in emotion_mapping:
                    for key, value in emotion_mapping[emotion].items():
                        requirements[key] += value
        
        # 장면 타입별 조정
        scene_adjustments = {
            "romantic": {"floral": 4, "warm": 3, "gourmand": 2, "mood": 0.5},
            "horror": {"dark": 6, "intensity": 5, "earthy": 3, "mood": -0.7},
            "action": {"spicy": 5, "intensity": 4, "dark": 2},
            "comedy": {"citrus": 3, "fresh": 4, "mood": 0.6},
            "drama": {"woody": 2, "complexity": 2},
            "thriller": {"dark": 4, "intensity": 4, "spicy": 2, "mood": -0.3},
            "fantasy": {"floral": 3, "woody": 3, "complexity": 3, "mood": 0.2},
            "sci-fi": {"marine": 3, "intensity": 2, "complexity": 3}
        }
        
        if scene_type in scene_adjustments:
            for key, value in scene_adjustments[scene_type].items():
                requirements[key] += value
        
        # 추가 키워드 분석
        additional_keywords = {
            "사랑": {"floral": 3, "warm": 2, "mood": 0.4},
            "이별": {"earthy": 3, "dark": 2, "mood": -0.5},
            "만남": {"fresh": 3, "citrus": 2, "mood": 0.3},
            "추억": {"woody": 2, "earthy": 1, "mood": 0.1},
            "꿈": {"floral": 2, "fresh": 2, "mood": 0.3},
            "희망": {"citrus": 3, "fresh": 3, "mood": 0.5},
            "절망": {"dark": 4, "earthy": 3, "mood": -0.6},
            "불": {"spicy": 4, "warm": 5, "intensity": 3},
            "얼음": {"fresh": 5, "marine": 3, "intensity": -1},
            "어둠": {"dark": 5, "intensity": 2, "mood": -0.4},
            "빛": {"citrus": 3, "fresh": 3, "mood": 0.4}
        }
        
        for keyword, effects in additional_keywords.items():
            if keyword in desc:
                for key, value in effects.items():
                    requirements[key] += value
        
        # 값 정규화 및 제한
        for key in requirements:
            if key == "mood":
                requirements[key] = max(-1.0, min(1.0, requirements[key]))
            else:
                requirements[key] = max(0.0, min(10.0, requirements[key]))
        
        return requirements

    def calculate_note_compatibility(self, note: ScentNote, requirements: Dict[str, float]) -> float:
        """노트와 요구사항 간의 호환성 점수 계산"""
        score = 0.0
        
        # 카테고리 매칭 (40% 가중치)
        category_score = requirements.get(note.category, 0) / 10.0
        score += category_score * 0.4
        
        # 무드 매칭 (30% 가중치)
        mood_diff = abs(note.mood_score - requirements["mood"])
        mood_score = 1.0 - mood_diff
        score += mood_score * 0.3
        
        # 강도 매칭 (20% 가중치)
        intensity_diff = abs(note.intensity - requirements["intensity"]) / 10.0
        intensity_score = 1.0 - intensity_diff
        score += intensity_score * 0.2
        
        # 복잡성 고려 (10% 가중치)
        complexity_bonus = min(requirements["complexity"] / 10.0, 1.0)
        score += complexity_bonus * 0.1
        
        return max(0.0, min(1.0, score))

    def generate_optimized_composition(self, requirements: Dict[str, float]) -> Tuple[List, List, List, float]:
        """최적화된 향 조합 생성"""
        
        # 각 레이어별 노트 선택
        top_notes = []
        middle_notes = []
        base_notes = []
        
        # 탑노트 선택 (2-4개, 신선함과 첫인상)
        top_candidates = []
        for note in self.scent_database["top"]:
            score = self.calculate_note_compatibility(note, requirements)
            top_candidates.append((note, score))
        
        top_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 후보 중에서 다양성 고려하여 선택
        selected_categories = set()
        for note, score in top_candidates:
            if score > 0.4 and len(top_notes) < 4:
                if note.category not in selected_categories or len(top_notes) < 2:
                    top_notes.append(note)
                    selected_categories.add(note.category)
        
        # 미들노트 선택 (3-6개, 주요 특성)
        middle_candidates = []
        for note in self.scent_database["middle"]:
            score = self.calculate_note_compatibility(note, requirements)
            middle_candidates.append((note, score))
        
        middle_candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected_categories = set()
        for note, score in middle_candidates:
            if score > 0.3 and len(middle_notes) < 6:
                if note.category not in selected_categories or len(middle_notes) < 3:
                    middle_notes.append(note)
                    selected_categories.add(note.category)
        
        # 베이스노트 선택 (2-4개, 지속성과 깊이)
        base_candidates = []
        for note in self.scent_database["base"]:
            score = self.calculate_note_compatibility(note, requirements)
            base_candidates.append((note, score))
        
        base_candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected_categories = set()
        for note, score in base_candidates:
            if score > 0.4 and len(base_notes) < 4:
                if note.category not in selected_categories or len(base_notes) < 2:
                    base_notes.append(note)
                    selected_categories.add(note.category)
        
        # 전체 조화 점수 계산
        harmony_score = self._calculate_harmony(top_notes, middle_notes, base_notes, requirements)
        
        return top_notes, middle_notes, base_notes, harmony_score

    def _calculate_harmony(self, top_notes: List, middle_notes: List, base_notes: List, 
                          requirements: Dict[str, float]) -> float:
        """조합의 전체적 조화 점수"""
        all_notes = top_notes + middle_notes + base_notes
        
        if not all_notes:
            return 0.0
        
        # 강도 균형
        intensities = [note.intensity for note in all_notes]
        avg_intensity = sum(intensities) / len(intensities)
        intensity_balance = 1.0 - abs(avg_intensity - requirements["intensity"]) / 10.0
        
        # 무드 일관성
        moods = [note.mood_score for note in all_notes]
        avg_mood = sum(moods) / len(moods)
        mood_consistency = 1.0 - abs(avg_mood - requirements["mood"])
        
        # 카테고리 다양성
        categories = set(note.category for note in all_notes)
        diversity = min(len(categories) / 6.0, 1.0)  # 최대 6개 카테고리
        
        # 레이어 균형
        layer_balance = 1.0
        if len(top_notes) == 0 or len(middle_notes) == 0 or len(base_notes) == 0:
            layer_balance = 0.5
        
        # 최종 조화 점수
        harmony = (intensity_balance * 0.3 + 
                  mood_consistency * 0.3 + 
                  diversity * 0.2 + 
                  layer_balance * 0.2)
        
        return max(0.0, min(1.0, harmony))

    def simulate_scent_development(self, top_notes: List, middle_notes: List, base_notes: List) -> Dict:
        """향의 시간별 전개 시뮬레이션"""
        timeline = {
            "0-15분": {"notes": top_notes, "strength": 100},
            "15분-2시간": {"notes": top_notes + middle_notes, "strength": 85},
            "2-6시간": {"notes": middle_notes, "strength": 70},
            "6-12시간": {"notes": middle_notes + base_notes, "strength": 50},
            "12시간+": {"notes": base_notes, "strength": 30}
        }
        
        development = {}
        for time_period, data in timeline.items():
            notes_info = []
            for note in data["notes"]:
                notes_info.append({
                    "name": note.name,
                    "category": note.category,
                    "relative_strength": min(100, (note.intensity / 10.0) * data["strength"])
                })
            development[time_period] = {
                "notes": notes_info,
                "overall_strength": data["strength"]
            }
        
        return development

    def run_advanced_simulation(self, description: str, scene_type: str, 
                               emotions: List[str] = None, iterations: int = 500) -> Dict:
        """고급 시뮬레이션 실행"""
        print(f"향 조합 시뮬레이션 시작: {iterations}회")
        
        requirements = self.analyze_scene_comprehensive(description, scene_type, emotions)
        
        best_composition = None
        best_score = -1
        simulation_data = []
        
        for i in range(iterations):
            # 약간의 랜덤 변이 추가
            varied_req = requirements.copy()
            for key in varied_req:
                if key != "mood":
                    noise = random.uniform(-0.8, 0.8)
                    varied_req[key] = max(0, min(10, varied_req[key] + noise))
                else:
                    noise = random.uniform(-0.15, 0.15)
                    varied_req[key] = max(-1, min(1, varied_req[key] + noise))
            
            top_notes, middle_notes, base_notes, harmony = self.generate_optimized_composition(varied_req)
            
            # 종합 점수 계산
            all_notes = top_notes + middle_notes + base_notes
            if all_notes:
                avg_intensity = sum(note.intensity for note in all_notes) / len(all_notes)
                avg_mood = sum(note.mood_score for note in all_notes) / len(all_notes)
                
                intensity_match = 1.0 - abs(avg_intensity - requirements["intensity"]) / 10.0
                mood_match = 1.0 - abs(avg_mood - requirements["mood"])
                
                total_score = (harmony * 0.4 + intensity_match * 0.3 + mood_match * 0.3)
                
                if total_score > best_score:
                    best_score = total_score
                    best_composition = {
                        "top_notes": top_notes,
                        "middle_notes": middle_notes,
                        "base_notes": base_notes,
                        "harmony_score": harmony,
                        "intensity_match": intensity_match,
                        "mood_match": mood_match,
                        "total_score": total_score
                    }
                
                # 시뮬레이션 데이터 수집
                if i % 50 == 0:
                    simulation_data.append({
                        "iteration": i,
                        "score": total_score,
                        "harmony": harmony,
                        "intensity_match": intensity_match,
                        "mood_match": mood_match
                    })
        
        if not best_composition:
            return {"error": "적합한 조합을 찾을 수 없습니다"}
        
        # 시간별 전개 시뮬레이션
        development = self.simulate_scent_development(
            best_composition["top_notes"],
            best_composition["middle_notes"], 
            best_composition["base_notes"]
        )
        
        # 제조 공식 생성
        formula = self._generate_detailed_formula(best_composition)
        
        result = {
            "composition": {
                "top_notes": [self._note_to_dict(note) for note in best_composition["top_notes"]],
                "middle_notes": [self._note_to_dict(note) for note in best_composition["middle_notes"]],
                "base_notes": [self._note_to_dict(note) for note in best_composition["base_notes"]],
                "overall_intensity": sum(note.intensity for note in 
                                       best_composition["top_notes"] + 
                                       best_composition["middle_notes"] + 
                                       best_composition["base_notes"]) / len(
                                       best_composition["top_notes"] + 
                                       best_composition["middle_notes"] + 
                                       best_composition["base_notes"]),
                "mood_match": best_composition["mood_match"],
                "harmony_score": best_composition["harmony_score"],
                "formula": formula
            },
            "development_timeline": development,
            "simulation_results": {
                "iterations": iterations,
                "best_score": best_score,
                "convergence_data": simulation_data
            },
            "scene_analysis": {
                "requirements": requirements,
                "detected_emotions": emotions or [],
                "scene_type": scene_type
            }
        }
        
        print(f"시뮬레이션 완료. 최고 점수: {best_score:.3f}")
        return result

    def _note_to_dict(self, note: ScentNote) -> Dict:
        """노트를 딕셔너리로 변환"""
        return {
            "name": note.name,
            "category": note.category,
            "intensity": note.intensity,
            "longevity": note.longevity,
            "volatility": note.volatility,
            "mood_score": note.mood_score
        }

    def _generate_detailed_formula(self, composition: Dict) -> Dict:
        """향료 원료 기반 상세 제조 공식 생성"""
        top_notes = composition["top_notes"]
        middle_notes = composition["middle_notes"]
        base_notes = composition["base_notes"]
        
        formula = {
            "원료별 정확한 농도": {},
            "원료 추출 정보": {},
            "혼합 순서": [],
            "희석 비율": {},
            "화학적 안정성": {},
            "품질 관리": {}
        }
        
        # 각 원료의 정확한 농도 계산 (mg/ml 기준)
        all_notes = top_notes + middle_notes + base_notes
        total_intensity = sum(note.intensity for note in all_notes) if all_notes else 1
        
        for note in all_notes:
            # 실제 향료 농도 (mg/ml)
            concentration_mgml = (note.intensity / total_intensity) * 50  # 50mg/ml 기준
            
            formula["원료별 정확한 농도"][note.name] = {
                "농도": f"{concentration_mgml:.2f} mg/ml",
                "백분율": f"{(note.intensity/total_intensity)*100:.2f}%",
                "휘발성": note.volatility,
                "지속시간": f"{note.longevity:.1f}시간",
                "카테고리": note.category
            }
        
        # 원료 추출 정보 및 공급원
        extraction_methods = {
            "citrus": {"method": "Cold Press", "yield": "0.3-0.5%", "origin": "과피"},
            "floral": {"method": "Steam Distillation", "yield": "0.1-0.3%", "origin": "꽃잎"},
            "woody": {"method": "Steam Distillation", "yield": "1-3%", "origin": "목재"},
            "herbal": {"method": "Steam Distillation", "yield": "0.5-2%", "origin": "잎/줄기"},
            "spicy": {"method": "Steam Distillation", "yield": "1-4%", "origin": "씨앗/뿌리"},
            "marine": {"method": "Synthetic", "yield": "99%+", "origin": "화학합성"},
            "earthy": {"method": "Solvent Extraction", "yield": "5-15%", "origin": "뿌리/토양"},
            "gourmand": {"method": "CO2 Extraction", "yield": "8-20%", "origin": "콩/바닐라"},
            "fruity": {"method": "Steam Distillation", "yield": "0.2-1%", "origin": "과육"},
            "smoky": {"method": "Pyrolysis", "yield": "Variable", "origin": "탄화물질"}
        }
        
        for note in all_notes:
            category = note.category
            if category in extraction_methods:
                formula["원료 추출 정보"][note.name] = extraction_methods[category]
        
        # 화학적으로 정확한 혼합 순서
        base_notes_sorted = sorted(base_notes, key=lambda x: x.longevity, reverse=True)
        middle_notes_sorted = sorted(middle_notes, key=lambda x: x.intensity, reverse=True)
        top_notes_sorted = sorted(top_notes, key=lambda x: x.intensity, reverse=True)
        
        formula["혼합 순서"] = [
            f"1단계: 최고 지속성 베이스노트부터 - {', '.join([n.name for n in base_notes_sorted])}",
            f"2단계: 강도순 미들노트 추가 - {', '.join([n.name for n in middle_notes_sorted])}",
            f"3단계: 휘발성 탑노트 마지막 - {', '.join([n.name for n in top_notes_sorted])}",
            "4단계: 30분간 교반하며 분자 결합 유도",
            "5단계: 96% 에탄올로 희석 (3:1 비율)",
            "6단계: 증류수 최종 조정 (전체의 5%)"
        ]
        
        # 정확한 희석 비율
        total_oils = len(all_notes)
        oil_percentage = min(25, max(5, total_oils * 2))  # 원료 수에 따른 오일 농도
        
        formula["희석 비율"] = {
            "에센셜 오일": f"{oil_percentage}%",
            "에탄올 (96%)": f"{85-oil_percentage}%", 
            "증류수": "5%",
            "안정제": "2% (Dipropylene glycol)",
            "기준 용량": "100ml"
        }
        
        # 화학적 안정성 정보
        formula["화학적 안정성"] = {
            "pH 범위": "6.5-7.5 (중성 유지)",
            "산화 방지": "BHT 0.01% 또는 Vitamin E 0.02%",
            "미생물 억제": "Phenoxyethanol 0.5%",
            "색변화 방지": "자외선 차단 용기 필수",
            "휘발 억제": "밀폐 저장, 15-20°C"
        }
        
        # 품질 관리 기준
        avg_longevity = sum(note.longevity for note in all_notes) / len(all_notes) if all_notes else 6
        aging_period = max(3, min(12, int(avg_longevity)))
        
        formula["품질 관리"] = {
            "숙성 기간": f"{aging_period}주",
            "저장 온도": "15-20°C ±2°C",
            "상대습도": "50-60%",
            "용기 재질": "코발트 블루 유리 또는 갈색 유리",
            "품질 검사": "주 1회 관능평가, 월 1회 가스크로마토그래피",
            "유통기한": "제조일로부터 3년 (개봉 후 1년)"
        }
        
        return formula

# 독립 실행 가능한 함수들
def create_movie_scent(description: str, scene_type: str = "drama", 
                      emotions: List[str] = None) -> Dict:
    """영화 장면 기반 향수 생성"""
    simulator = StandaloneScentSimulator()
    return simulator.run_advanced_simulation(description, scene_type, emotions, 200)

if __name__ == "__main__":
    # 테스트 실행
    test_description = "비 오는 밤 옥상에서 마지막 이별을 고하는 장면"
    result = create_movie_scent(test_description, "drama", ["sad", "melancholy"])
    print(json.dumps(result, ensure_ascii=False, indent=2))