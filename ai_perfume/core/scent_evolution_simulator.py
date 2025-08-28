from typing import Dict, List
import numpy as np
from datetime import datetime, timedelta

class ScentEvolutionSimulator:
    def __init__(self):
        # 향료 특성 정의
        self.note_properties = {
            # 휘발성(1-10), 강도(1-10), 지속시간(시간)
            'top_notes': {
                '베르가못': {'volatility': 9, 'intensity': 7, 'duration': 2},
                '레몬': {'volatility': 9, 'intensity': 6, 'duration': 1.5},
                '라임': {'volatility': 9, 'intensity': 6, 'duration': 1.5},
                '오렌지': {'volatility': 8, 'intensity': 7, 'duration': 2},
                '민트': {'volatility': 8, 'intensity': 7, 'duration': 2},
                '유칼립투스': {'volatility': 8, 'intensity': 8, 'duration': 2.5}
            },
            'middle_notes': {
                '장미': {'volatility': 6, 'intensity': 8, 'duration': 4},
                '자스민': {'volatility': 6, 'intensity': 8, 'duration': 5},
                '일랑일랑': {'volatility': 5, 'intensity': 7, 'duration': 5},
                '라벤더': {'volatility': 7, 'intensity': 6, 'duration': 3},
                '제라늄': {'volatility': 6, 'intensity': 6, 'duration': 4},
                '네롤리': {'volatility': 7, 'intensity': 7, 'duration': 3}
            },
            'base_notes': {
                '바닐라': {'volatility': 3, 'intensity': 8, 'duration': 12},
                '샌달우드': {'volatility': 3, 'intensity': 7, 'duration': 24},
                '파촐리': {'volatility': 2, 'intensity': 8, 'duration': 24},
                '머스크': {'volatility': 2, 'intensity': 7, 'duration': 48},
                '앰버': {'volatility': 3, 'intensity': 8, 'duration': 24},
                '시더우드': {'volatility': 4, 'intensity': 6, 'duration': 12}
            }
        }

    def simulate_evolution(self, recipe: Dict, duration_hours: int = 24) -> Dict:
        """향의 시간에 따른 발현과 변화 시뮬레이션"""
        # 시간대별 강도 계산
        timeline = []
        start_time = datetime.now()
        
        for hour in range(duration_hours + 1):
            current_time = start_time + timedelta(hours=hour)
            
            # 각 노트별 강도 계산
            intensities = {
                'top_notes': self._calculate_note_intensity('top_notes', recipe['top_notes'], hour),
                'middle_notes': self._calculate_note_intensity('middle_notes', recipe['middle_notes'], hour),
                'base_notes': self._calculate_note_intensity('base_notes', recipe['base_notes'], hour)
            }
            
            # 전체 향 프로파일 계산
            total_intensity = sum([
                np.mean(list(intensities[note_type].values()))
                for note_type in intensities
            ]) / 3
            
            # 주요 향 특성 파악
            dominant_notes = self._get_dominant_notes(intensities)
            
            timeline.append({
                'time': current_time.strftime('%H:%M'),
                'hour': hour,
                'intensities': intensities,
                'total_intensity': total_intensity,
                'dominant_notes': dominant_notes
            })
        
        return {
            'timeline': timeline,
            'longevity_score': self._calculate_longevity_score(timeline),
            'sillage_score': self._calculate_sillage_score(timeline),
            'evolution_quality': self._evaluate_evolution_quality(timeline)
        }
    
    def _calculate_note_intensity(self, note_type: str, notes: List[str], hour: int) -> Dict[str, float]:
        """특정 시점의 각 노트 강도 계산"""
        intensities = {}
        for note in notes:
            if note in self.note_properties[note_type]:
                props = self.note_properties[note_type][note]
                
                # 시간에 따른 강도 감소 계산 (지수 감소)
                decay_rate = np.log(2) / props['duration']  # 반감기 기반 감소율
                initial_intensity = props['intensity']
                
                # 발현 시간 고려 (top: 즉시, middle: 30분 후, base: 1시간 후)
                onset_delay = {
                    'top_notes': 0,
                    'middle_notes': 0.5,
                    'base_notes': 1
                }
                
                if hour < onset_delay[note_type]:
                    intensity = 0
                else:
                    adjusted_time = hour - onset_delay[note_type]
                    intensity = initial_intensity * np.exp(-decay_rate * adjusted_time)
                
                intensities[note] = max(0, intensity)
                
        return intensities
    
    def _get_dominant_notes(self, intensities: Dict) -> List[str]:
        """현재 시점에서 가장 강한 향 노트들 추출"""
        all_notes = []
        for note_type in intensities:
            for note, intensity in intensities[note_type].items():
                if intensity > 1.0:  # 역치 이상의 강도만 고려
                    all_notes.append((note, intensity))
        
        # 강도 기준 정렬
        all_notes.sort(key=lambda x: x[1], reverse=True)
        return [note for note, _ in all_notes[:3]]  # 상위 3개 노트
    
    def _calculate_longevity_score(self, timeline: List[Dict]) -> float:
        """향의 지속성 점수 계산"""
        # 마지막 25%의 시간대에서 평균 강도가 2 이상인 비율
        threshold = 2.0
        last_quarter = timeline[int(len(timeline) * 0.75):]
        above_threshold = sum(1 for t in last_quarter if t['total_intensity'] > threshold)
        
        return above_threshold / len(last_quarter) * 10  # 0-10 점수
    
    def _calculate_sillage_score(self, timeline: List[Dict]) -> float:
        """향의 확산력 점수 계산"""
        # 첫 6시간 동안의 평균 강도
        first_6_hours = timeline[:7]  # 0시간 포함
        avg_intensity = np.mean([t['total_intensity'] for t in first_6_hours])
        
        # 0-10 범위로 정규화
        return min(10, avg_intensity * 1.5)
    
    def _evaluate_evolution_quality(self, timeline: List[Dict]) -> Dict:
        """향의 발현 품질 평가"""
        # 이상적인 발현 곡선과 비교
        ideal_evolution = {
            'top_phase': {'duration': 2, 'min_intensity': 6},
            'middle_phase': {'duration': 6, 'min_intensity': 4},
            'base_phase': {'duration': 16, 'min_intensity': 2}
        }
        
        phase_scores = {}
        
        # Top 노트 평가 (0-2시간)
        top_phase = timeline[:3]
        phase_scores['top_phase'] = self._evaluate_phase(
            top_phase,
            ideal_evolution['top_phase']['min_intensity']
        )
        
        # Middle 노트 평가 (2-8시간)
        middle_phase = timeline[3:9]
        phase_scores['middle_phase'] = self._evaluate_phase(
            middle_phase,
            ideal_evolution['middle_phase']['min_intensity']
        )
        
        # Base 노트 평가 (8-24시간)
        base_phase = timeline[9:]
        phase_scores['base_phase'] = self._evaluate_phase(
            base_phase,
            ideal_evolution['base_phase']['min_intensity']
        )
        
        # 전체 품질 점수 계산
        overall_score = (
            phase_scores['top_phase'] * 0.3 +
            phase_scores['middle_phase'] * 0.4 +
            phase_scores['base_phase'] * 0.3
        )
        
        return {
            'phase_scores': phase_scores,
            'overall_score': overall_score,
            'rating': self._get_quality_rating(overall_score)
        }
    
    def _evaluate_phase(self, phase_timeline: List[Dict], min_intensity: float) -> float:
        """각 발현 단계의 품질 평가"""
        intensities = [t['total_intensity'] for t in phase_timeline]
        
        # 강도 충족도
        intensity_score = sum(1 for i in intensities if i >= min_intensity) / len(intensities)
        
        # 강도 변화의 부드러움
        smoothness = 1.0
        if len(intensities) > 1:
            changes = np.diff(intensities)
            smoothness = 1.0 - min(1.0, np.std(changes) / 2.0)
        
        return (intensity_score * 0.7 + smoothness * 0.3) * 10
    
    def _get_quality_rating(self, score: float) -> str:
        """품질 점수를 등급으로 변환"""
        if score >= 9.0:
            return "탁월함 (Excellent)"
        elif score >= 8.0:
            return "매우 좋음 (Very Good)"
        elif score >= 7.0:
            return "좋음 (Good)"
        elif score >= 6.0:
            return "보통 (Fair)"
        else:
            return "개선 필요 (Needs Improvement)"

    def get_improvement_suggestions(self, simulation_result: Dict) -> List[str]:
        """향 발현 개선을 위한 제안 생성"""
        suggestions = []
        phase_scores = simulation_result['evolution_quality']['phase_scores']
        
        # Top 노트 개선 제안
        if phase_scores['top_phase'] < 7.0:
            suggestions.append(
                "탑 노트의 강도가 약합니다. 시트러스나 민트 계열의 향료를 추가하거나 비율을 높이는 것을 고려하세요."
            )
        
        # Middle 노트 개선 제안
        if phase_scores['middle_phase'] < 7.0:
            suggestions.append(
                "미들 노트의 지속성이 부족합니다. 플로럴 계열의 향료를 보강하거나, 지속성이 좋은 우디 계열을 추가하세요."
            )
        
        # Base 노트 개선 제안
        if phase_scores['base_phase'] < 7.0:
            suggestions.append(
                "베이스 노트가 약합니다. 머스크나 앰버 계열의 향료를 추가하여 지속성을 높이세요."
            )
        
        # 전반적인 발현 곡선 개선 제안
        if simulation_result['longevity_score'] < 7.0:
            suggestions.append(
                "전반적인 지속성이 부족합니다. 고정제(피克斯아티브)의 비율을 높이는 것을 고려하세요."
            )
        
        if simulation_result['sillage_score'] < 7.0:
            suggestions.append(
                "확산력이 약합니다. 휘발성이 높은 향료의 비율을 조정하거나, 알코올 농도를 높이는 것을 고려하세요."
            )
        
        return suggestions 