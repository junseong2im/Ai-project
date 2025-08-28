from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

class PerfumeNotesManager:
    """향수 노트 관리 클래스"""
    
    def __init__(self, database_path: str | Path = "data/perfume_notes_database.json") -> None:
        self.database_path = Path(database_path)
        self.database = self._load_database()
        
    def _load_database(self) -> Dict[str, Any]:
        """향수 원료 데이터베이스 로드"""
        if self.database_path.exists():
            with self.database_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        raise FileNotFoundError(f"데이터베이스 파일을 찾을 수 없습니다: {self.database_path}")
    
    def get_note_info(self, note_name: str) -> Optional[Dict[str, Any]]:
        """특정 향료의 정보 조회"""
        for category in ['top_notes', 'middle_notes', 'base_notes']:
            if category in self.database.get('notes_database', {}):
                for family in self.database['notes_database'][category].values():
                    for note in family:
                        if note.get('name') == note_name:
                            return note
        return None
    
    def get_compatible_notes(self, note_name: str) -> List[str]:
        """특정 향료와 잘 어울리는 향료 목록 반환"""
        note_info = self.get_note_info(note_name)
        if note_info and 'common_combinations' in note_info:
            return note_info['common_combinations']
        return []
    
    def get_seasonal_notes(self, season: str) -> Dict[str, List[str]]:
        """계절에 어울리는 향료 목록 반환"""
        seasonal_notes: Dict[str, List[str]] = {'top': [], 'middle': [], 'base': []}
        
        for category in ['top_notes', 'middle_notes', 'base_notes']:
            category_key = category.split('_')[0]
            if category in self.database.get('notes_database', {}):
                for family in self.database['notes_database'][category].values():
                    for note in family:
                        if 'season_affinity' in note and season in note['season_affinity']:
                            seasonal_notes[category_key].append(note['name'])
        
        return seasonal_notes
    
    def get_mood_notes(self, mood: str) -> List[str]:
        """특정 감정/분위기에 어울리는 향료 목록 반환"""
        mood_notes: List[str] = []
        
        for category in ['top_notes', 'middle_notes', 'base_notes']:
            if category in self.database.get('notes_database', {}):
                for family in self.database['notes_database'][category].values():
                    for note in family:
                        if 'mood_effects' in note and mood in note['mood_effects']:
                            mood_notes.append(note['name'])
        
        return mood_notes
    
    def get_concentration_guidelines(self, concentration_type: str) -> Optional[Dict[str, Any]]:
        """향수 농도별 가이드라인 반환"""
        guidelines = self.database.get('concentration_guidelines', {})
        return guidelines.get(concentration_type)
    
    def get_classic_combinations(self) -> List[Dict[str, Any]]:
        """클래식한 향료 조합 목록 반환"""
        combinations = self.database.get('note_combinations', {})
        return combinations.get('classic', [])
    
    def get_modern_combinations(self) -> List[Dict[str, Any]]:
        """현대적인 향료 조합 목록 반환"""
        combinations = self.database.get('note_combinations', {})
        return combinations.get('modern', [])
    
    def suggest_combination(self, mood: str, season: str) -> Dict[str, List[str]]:
        """감정과 계절에 맞는 향료 조합 추천"""
        # 계절에 맞는 향료 선택
        seasonal_notes = self.get_seasonal_notes(season)
        
        # 감정에 맞는 향료 선택
        mood_notes = set(self.get_mood_notes(mood))
        
        # 향료 선택
        combination: Dict[str, List[str]] = {
            'top_notes': [],
            'middle_notes': [],
            'base_notes': []
        }
        
        # 각 노트 카테고리별로 2-3개 선택
        for category in combination.keys():
            category_notes = seasonal_notes[category.split('_')[0]]
            # 감정에 맞는 향료 우선 선택
            mood_category_notes = list(set(category_notes) & mood_notes)
            
            if mood_category_notes:
                selected_count = min(2, len(mood_category_notes))
                combination[category].extend(random.sample(mood_category_notes, selected_count))
            
            # 부족한 만큼 일반 향료에서 선택
            remaining = 2 - len(combination[category])
            if remaining > 0 and category_notes:
                available_notes = [note for note in category_notes if note not in combination[category]]
                if available_notes:
                    additional_count = min(remaining, len(available_notes))
                    additional_notes = random.sample(available_notes, additional_count)
                    combination[category].extend(additional_notes)
        
        return combination
    
    def calculate_concentration(
        self, 
        notes: Dict[str, List[str]], 
        concentration_type: str = "eau_de_parfum"
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """향료 농도 계산"""
        guidelines = self.get_concentration_guidelines(concentration_type)
        if not guidelines:
            return None
            
        # 농도 범위에서 최소값 사용
        concentration_str = guidelines.get('concentration', '15-20%')
        total_concentration = float(concentration_str.split('-')[0])
        note_ratios = guidelines.get('note_ratio', {})
        
        concentrations: Dict[str, Dict[str, float]] = {
            'top_notes': {},
            'middle_notes': {},
            'base_notes': {}
        }
        
        for category in notes:
            category_key = category.split('_')[0]
            if category_key in note_ratios:
                ratio_str = note_ratios[category_key]
                category_ratio = float(ratio_str.split('-')[0]) / 100
                note_count = len(notes[category])
                
                if note_count > 0:
                    per_note_concentration = (total_concentration * category_ratio) / note_count
                    
                    for note in notes[category]:
                        note_info = self.get_note_info(note)
                        if note_info and 'intensity' in note_info:
                            # 향료의 강도에 따라 농도 조절
                            intensity_factor = note_info['intensity'] / 10
                            final_concentration = per_note_concentration * intensity_factor
                            concentrations[category][note] = round(final_concentration, 2)
                        else:
                            # 기본 농도 사용
                            concentrations[category][note] = round(per_note_concentration, 2)
        
        return concentrations 