#!/usr/bin/env python3
"""
향 시뮬레이션 시스템
영화 장면 분석을 통한 향 조합 생성 및 최적화
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time
import logging

logger = logging.getLogger(__name__)

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

class ScentComposition:
    """향 조합"""
    def __init__(self):
        self.top_notes = []  # 탑노트 (휘발성 높음)
        self.middle_notes = []  # 미들노트 (중간)
        self.base_notes = []  # 베이스노트 (지속성 높음)
        self.overall_intensity = 0.0
        self.mood_match = 0.0
        self.complexity = 0.0
        self.harmony_score = 0.0

class ScentSimulator:
    """향 시뮬레이션 및 최적화 엔진"""
    
    def __init__(self):
        self.scent_database = self._initialize_scent_database()
        self.simulation_iterations = 1000  # 시뮬레이션 반복 횟수
        
    def _initialize_scent_database(self) -> Dict[str, List[ScentNote]]:
        """향료 데이터베이스 초기화"""
        database = {
            "top": [
                # 상쾌한 탑노트들
                ScentNote("베르가못", "citrus", 8.0, 2.0, "top", 0.7),
                ScentNote("레몬", "citrus", 9.0, 1.5, "top", 0.8),
                ScentNote("페퍼민트", "herbal", 8.5, 2.5, "top", 0.6),
                ScentNote("라벤더", "herbal", 6.0, 4.0, "top", 0.4),
                ScentNote("유칼립투스", "herbal", 9.0, 3.0, "top", 0.3),
                ScentNote("그린 애플", "fruity", 7.0, 2.0, "top", 0.7),
                ScentNote("바다 소금", "marine", 6.0, 3.0, "top", 0.2),
                ScentNote("오존", "marine", 7.0, 1.5, "top", 0.5),
                ScentNote("핑크 페퍼", "spicy", 7.5, 3.0, "top", 0.1),
                ScentNote("진저", "spicy", 8.0, 2.5, "top", 0.3),
            ],
            "middle": [
                # 풍부한 미들노트들
                ScentNote("로즈", "floral", 8.0, 6.0, "middle", 0.8),
                ScentNote("자스민", "floral", 9.0, 7.0, "middle", 0.7),
                ScentNote("릴리", "floral", 6.0, 5.0, "middle", 0.6),
                ScentNote("제라늄", "floral", 7.0, 6.0, "middle", 0.5),
                ScentNote("시나몬", "spicy", 8.0, 5.0, "middle", 0.2),
                ScentNote("클로브", "spicy", 9.0, 6.0, "middle", -0.1),
                ScentNote("바닐라", "gourmand", 7.0, 8.0, "middle", 0.6),
                ScentNote("코코넛", "gourmand", 6.0, 5.0, "middle", 0.7),
                ScentNote("티 리프", "green", 5.0, 4.0, "middle", 0.3),
                ScentNote("바이올렛", "floral", 5.5, 4.5, "middle", 0.4),
            ],
            "base": [
                # 깊이 있는 베이스노트들
                ScentNote("샌달우드", "woody", 7.0, 10.0, "base", 0.1),
                ScentNote("시더우드", "woody", 8.0, 9.0, "base", 0.0),
                ScentNote("머스크", "musk", 6.0, 10.0, "base", 0.2),
                ScentNote("앰버", "amber", 8.0, 9.5, "base", 0.3),
                ScentNote("바닐라", "gourmand", 7.0, 8.0, "base", 0.6),
                ScentNote("파출리", "earthy", 9.0, 10.0, "base", -0.2),
                ScentNote("오크모스", "earthy", 6.0, 9.0, "base", -0.1),
                ScentNote("베티버", "woody", 7.5, 9.5, "base", -0.3),
                ScentNote("통카빈", "gourmand", 8.0, 8.5, "base", 0.4),
                ScentNote("스모키", "smoky", 9.0, 9.0, "base", -0.4),
            ]
        }
        return database

    def analyze_scene_requirements(self, description: str, scene_type: str, 
                                 emotions: List[str] = None) -> Dict[str, float]:
        """영화 장면 분석하여 향료 요구사항 도출"""
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
            "dark": 0.0
        }
        
        description_lower = description.lower()
        
        # 장소별 요구사항
        if any(word in description_lower for word in ["바다", "해변", "파도", "물"]):
            requirements["marine"] = 8.0
            requirements["fresh"] = 7.0
            
        if any(word in description_lower for word in ["숲", "나무", "자연", "산"]):
            requirements["woody"] = 8.0
            requirements["fresh"] = 6.0
            
        if any(word in description_lower for word in ["꽃", "정원", "봄"]):
            requirements["floral"] = 8.0
            requirements["mood"] = 0.6
            
        if any(word in description_lower for word in ["밤", "어둠", "무서운"]):
            requirements["dark"] = 7.0
            requirements["intensity"] = 8.0
            requirements["mood"] = -0.5
            
        if any(word in description_lower for word in ["따뜻한", "햇살", "여름"]):
            requirements["warm"] = 7.0
            requirements["mood"] = 0.5
            
        # 감정별 요구사항
        if emotions:
            for emotion in emotions:
                if emotion in ["love", "happy", "joy"]:
                    requirements["mood"] += 0.3
                    requirements["floral"] += 2.0
                elif emotion in ["sad", "melancholy"]:
                    requirements["mood"] -= 0.4
                    requirements["dark"] += 3.0
                elif emotion in ["fear", "tension"]:
                    requirements["dark"] += 4.0
                    requirements["intensity"] += 2.0
                elif emotion in ["anger", "rage"]:
                    requirements["spicy"] += 3.0
                    requirements["intensity"] += 3.0
                    
        # 장면 타입별 조정
        if scene_type == "romantic":
            requirements["floral"] += 3.0
            requirements["warm"] += 2.0
            requirements["mood"] += 0.4
        elif scene_type == "horror":
            requirements["dark"] += 5.0
            requirements["intensity"] += 4.0
            requirements["mood"] -= 0.6
        elif scene_type == "action":
            requirements["spicy"] += 4.0
            requirements["intensity"] += 3.0
        elif scene_type == "comedy":
            requirements["fresh"] += 3.0
            requirements["mood"] += 0.5
            
        # 값 정규화
        for key in requirements:
            if key != "mood":
                requirements[key] = max(0, min(10, requirements[key]))
            else:
                requirements[key] = max(-1, min(1, requirements[key]))
                
        return requirements

    def generate_composition(self, requirements: Dict[str, float]) -> ScentComposition:
        """요구사항에 따른 향 조합 생성"""
        composition = ScentComposition()
        
        # 노트 선택 확률 계산
        def calculate_note_score(note: ScentNote, req: Dict[str, float]) -> float:
            score = 0.0
            
            # 카테고리별 점수
            if note.category in req:
                score += req[note.category] * 0.3
                
            # 무드 매칭
            mood_diff = abs(note.mood_score - req["mood"])
            score += (1.0 - mood_diff) * 0.4
            
            # 강도 매칭
            intensity_diff = abs(note.intensity - req["intensity"]) / 10.0
            score += (1.0 - intensity_diff) * 0.3
            
            return max(0, score)
        
        # 탑노트 선택 (2-4개)
        top_candidates = [(note, calculate_note_score(note, requirements)) 
                         for note in self.scent_database["top"]]
        top_candidates.sort(key=lambda x: x[1], reverse=True)
        
        num_top = random.randint(2, 4)
        for i in range(min(num_top, len(top_candidates))):
            if top_candidates[i][1] > 0.3:  # 최소 점수 기준
                composition.top_notes.append(top_candidates[i][0])
        
        # 미들노트 선택 (2-5개)
        middle_candidates = [(note, calculate_note_score(note, requirements)) 
                            for note in self.scent_database["middle"]]
        middle_candidates.sort(key=lambda x: x[1], reverse=True)
        
        num_middle = random.randint(2, 5)
        for i in range(min(num_middle, len(middle_candidates))):
            if middle_candidates[i][1] > 0.3:
                composition.middle_notes.append(middle_candidates[i][0])
        
        # 베이스노트 선택 (2-4개)
        base_candidates = [(note, calculate_note_score(note, requirements)) 
                          for note in self.scent_database["base"]]
        base_candidates.sort(key=lambda x: x[1], reverse=True)
        
        num_base = random.randint(2, 4)
        for i in range(min(num_base, len(base_candidates))):
            if base_candidates[i][1] > 0.3:
                composition.base_notes.append(base_candidates[i][0])
        
        # 조합 평가
        self._evaluate_composition(composition, requirements)
        
        return composition

    def _evaluate_composition(self, composition: ScentComposition, 
                            requirements: Dict[str, float]) -> None:
        """향 조합 평가"""
        all_notes = composition.top_notes + composition.middle_notes + composition.base_notes
        
        if not all_notes:
            return
            
        # 전체 강도 계산
        composition.overall_intensity = np.mean([note.intensity for note in all_notes])
        
        # 무드 매칭 점수
        avg_mood = np.mean([note.mood_score for note in all_notes])
        composition.mood_match = 1.0 - abs(avg_mood - requirements["mood"])
        
        # 복잡성 점수
        composition.complexity = len(set(note.category for note in all_notes))
        
        # 하모니 점수 (카테고리 균형)
        categories = [note.category for note in all_notes]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
        # 너무 한 카테고리에 치우치지 않으면 좋은 하모니
        max_count = max(category_counts.values()) if category_counts else 1
        composition.harmony_score = 1.0 - (max_count / len(all_notes))

    def simulate_best_composition(self, description: str, scene_type: str, 
                                emotions: List[str] = None, 
                                iterations: int = None) -> Tuple[ScentComposition, List[Dict]]:
        """시뮬레이션을 통한 최적 향 조합 찾기"""
        if iterations is None:
            iterations = self.simulation_iterations
            
        requirements = self.analyze_scene_requirements(description, scene_type, emotions)
        
        best_composition = None
        best_score = -1
        simulation_results = []
        
        logger.info(f"향 조합 시뮬레이션 시작: {iterations}회 반복")
        
        for i in range(iterations):
            # 약간의 랜덤성을 추가하여 다양한 조합 시도
            varied_requirements = requirements.copy()
            for key in varied_requirements:
                if key != "mood":
                    noise = random.uniform(-0.5, 0.5)
                    varied_requirements[key] = max(0, min(10, varied_requirements[key] + noise))
                else:
                    noise = random.uniform(-0.1, 0.1)
                    varied_requirements[key] = max(-1, min(1, varied_requirements[key] + noise))
            
            composition = self.generate_composition(varied_requirements)
            
            # 종합 점수 계산
            score = (composition.mood_match * 0.4 + 
                    composition.harmony_score * 0.3 +
                    (1.0 - abs(composition.overall_intensity - requirements["intensity"]) / 10.0) * 0.3)
            
            if score > best_score:
                best_score = score
                best_composition = composition
            
            # 시뮬레이션 결과 기록
            if i % 100 == 0:
                simulation_results.append({
                    "iteration": i,
                    "score": score,
                    "intensity": composition.overall_intensity,
                    "mood_match": composition.mood_match,
                    "harmony": composition.harmony_score,
                    "complexity": composition.complexity
                })
        
        logger.info(f"시뮬레이션 완료. 최고 점수: {best_score:.3f}")
        
        return best_composition, simulation_results

    def composition_to_dict(self, composition: ScentComposition) -> Dict:
        """향 조합을 딕셔너리로 변환"""
        def note_to_dict(note: ScentNote) -> Dict:
            return {
                "name": note.name,
                "category": note.category,
                "intensity": note.intensity,
                "longevity": note.longevity,
                "volatility": note.volatility,
                "mood_score": note.mood_score
            }
        
        return {
            "top_notes": [note_to_dict(note) for note in composition.top_notes],
            "middle_notes": [note_to_dict(note) for note in composition.middle_notes],
            "base_notes": [note_to_dict(note) for note in composition.base_notes],
            "overall_intensity": composition.overall_intensity,
            "mood_match": composition.mood_match,
            "complexity": composition.complexity,
            "harmony_score": composition.harmony_score,
            "formula": self._generate_formula(composition)
        }
    
    def _generate_formula(self, composition: ScentComposition) -> Dict[str, str]:
        """향 조합의 제조 공식 생성"""
        formula = {
            "탑노트 (Top Notes)": [],
            "미들노트 (Middle Notes)": [], 
            "베이스노트 (Base Notes)": [],
            "조합 비율": "",
            "제조법": ""
        }
        
        # 각 노트별 비율 계산
        for note in composition.top_notes:
            ratio = f"{note.intensity * 2:.0f}%"
            formula["탑노트 (Top Notes)"].append(f"{note.name} ({ratio})")
            
        for note in composition.middle_notes:
            ratio = f"{note.intensity * 3:.0f}%"
            formula["미들노트 (Middle Notes)"].append(f"{note.name} ({ratio})")
            
        for note in composition.base_notes:
            ratio = f"{note.intensity * 2:.0f}%"
            formula["베이스노트 (Base Notes)"].append(f"{note.name} ({ratio})")
        
        # 전체 조합 비율
        total_top = len(composition.top_notes)
        total_middle = len(composition.middle_notes)  
        total_base = len(composition.base_notes)
        total = total_top + total_middle + total_base
        
        if total > 0:
            top_ratio = (total_top / total) * 100
            middle_ratio = (total_middle / total) * 100
            base_ratio = (total_base / total) * 100
            
            formula["조합 비율"] = f"탑노트 {top_ratio:.0f}% : 미들노트 {middle_ratio:.0f}% : 베이스노트 {base_ratio:.0f}%"
        
        # 제조법
        formula["제조법"] = (
            "1. 베이스노트를 먼저 블렌딩하여 기본 향을 만듭니다.\n"
            "2. 미들노트를 서서히 추가하며 조화롭게 섞습니다.\n"
            "3. 마지막으로 탑노트를 가볍게 블렌딩합니다.\n"
            "4. 24시간 이상 숙성시켜 향이 안정화되도록 합니다."
        )
        
        return formula

if __name__ == "__main__":
    # 테스트 코드
    simulator = ScentSimulator()
    
    test_description = "비 오는 밤 옥상에서 이별하는 장면"
    test_scene_type = "drama"
    test_emotions = ["sad", "melancholy"]
    
    composition, results = simulator.simulate_best_composition(
        test_description, test_scene_type, test_emotions, 500
    )
    
    result_dict = simulator.composition_to_dict(composition)
    print(json.dumps(result_dict, ensure_ascii=False, indent=2))