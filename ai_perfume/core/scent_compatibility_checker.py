from typing import List, Dict, Tuple
import numpy as np

class ScentCompatibilityChecker:
    def __init__(self):
        # 향료 패밀리 정의
        self.scent_families = {
            'citrus': ['베르가못', '레몬', '라임', '오렌지', '그레이프프룻', '만다린'],
            'floral': ['장미', '자스민', '일랑일랑', '라벤더', '제라늄', '네롤리'],
            'woody': ['샌달우드', '시더우드', '베티버', '파촐리', '구아이악우드'],
            'oriental': ['바닐라', '통카빈', '앰버', '벤조인', '미르'],
            'fougere': ['라벤더', '오크모스', '쿠마린', '제라늄'],
            'green': ['갈바넘', '바이올렛 리프', '민트', '바질'],
            'marine': ['씨솔트', '알게', '오존', '씨위드'],
            'spicy': ['시나몬', '클로브', '너트맥', '카다멈', '블랙페퍼']
        }

        # 향료 패밀리 간의 호환성 매트릭스 (0-1 범위, 1이 가장 호환성 좋음)
        self.family_compatibility = {
            'citrus': {
                'citrus': 0.8, 'floral': 0.9, 'woody': 0.7, 'oriental': 0.5,
                'fougere': 0.6, 'green': 0.8, 'marine': 0.7, 'spicy': 0.4
            },
            'floral': {
                'citrus': 0.9, 'floral': 0.7, 'woody': 0.8, 'oriental': 0.8,
                'fougere': 0.7, 'green': 0.6, 'marine': 0.5, 'spicy': 0.6
            },
            'woody': {
                'citrus': 0.7, 'floral': 0.8, 'woody': 0.8, 'oriental': 0.9,
                'fougere': 0.8, 'green': 0.6, 'marine': 0.5, 'spicy': 0.7
            },
            'oriental': {
                'citrus': 0.5, 'floral': 0.8, 'woody': 0.9, 'oriental': 0.8,
                'fougere': 0.6, 'green': 0.4, 'marine': 0.3, 'spicy': 0.8
            },
            'fougere': {
                'citrus': 0.6, 'floral': 0.7, 'woody': 0.8, 'oriental': 0.6,
                'fougere': 0.7, 'green': 0.8, 'marine': 0.6, 'spicy': 0.5
            },
            'green': {
                'citrus': 0.8, 'floral': 0.6, 'woody': 0.6, 'oriental': 0.4,
                'fougere': 0.8, 'green': 0.7, 'marine': 0.7, 'spicy': 0.5
            },
            'marine': {
                'citrus': 0.7, 'floral': 0.5, 'woody': 0.5, 'oriental': 0.3,
                'fougere': 0.6, 'green': 0.7, 'marine': 0.8, 'spicy': 0.4
            },
            'spicy': {
                'citrus': 0.4, 'floral': 0.6, 'woody': 0.7, 'oriental': 0.8,
                'fougere': 0.5, 'green': 0.5, 'marine': 0.4, 'spicy': 0.6
            }
        }

        # 향료별 특성 정의
        self.note_characteristics = {
            # 휘발성 (1-10), 강도 (1-10), 지속성 (1-10)
            'citrus': {'volatility': 8, 'intensity': 6, 'longevity': 4},
            'floral': {'volatility': 6, 'intensity': 7, 'longevity': 6},
            'woody': {'volatility': 4, 'intensity': 7, 'longevity': 8},
            'oriental': {'volatility': 3, 'intensity': 8, 'longevity': 9},
            'fougere': {'volatility': 5, 'intensity': 6, 'longevity': 7},
            'green': {'volatility': 7, 'intensity': 5, 'longevity': 5},
            'marine': {'volatility': 7, 'intensity': 5, 'longevity': 4},
            'spicy': {'volatility': 6, 'intensity': 8, 'longevity': 7}
        }

    def check_compatibility(self, notes: List[str]) -> Dict:
        """향료 조합의 호환성 검사"""
        if not notes:
            return {'score': 0, 'issues': ['향료가 지정되지 않았습니다.']}

        # 각 향료의 패밀리 찾기
        note_families = self._get_note_families(notes)
        if not note_families:
            return {'score': 0, 'issues': ['인식할 수 없는 향료가 포함되어 있습니다.']}

        # 호환성 점수 계산
        compatibility_scores = []
        issues = []

        # 모든 향료 쌍의 호환성 검사
        for i in range(len(note_families)):
            for j in range(i + 1, len(note_families)):
                family1, family2 = note_families[i], note_families[j]
                score = self.family_compatibility[family1][family2]
                compatibility_scores.append(score)

                if score < 0.5:
                    issues.append(f"{notes[i]}와(과) {notes[j]}의 호환성이 낮습니다. ({score:.2f})")

        # 전체 호환성 점수 계산
        overall_score = np.mean(compatibility_scores) if compatibility_scores else 0

        # 향 프로파일 분석
        profile = self._analyze_scent_profile(note_families)
        
        return {
            'score': overall_score,
            'issues': issues,
            'profile': profile
        }

    def _get_note_families(self, notes: List[str]) -> List[str]:
        """각 향료의 패밀리 찾기"""
        families = []
        for note in notes:
            for family, members in self.scent_families.items():
                if note in members:
                    families.append(family)
                    break
        return families

    def _analyze_scent_profile(self, families: List[str]) -> Dict:
        """향 프로파일 분석"""
        if not families:
            return {}

        # 각 특성의 평균 계산
        volatility = np.mean([self.note_characteristics[f]['volatility'] for f in families])
        intensity = np.mean([self.note_characteristics[f]['intensity'] for f in families])
        longevity = np.mean([self.note_characteristics[f]['longevity'] for f in families])

        return {
            'volatility': volatility,
            'intensity': intensity,
            'longevity': longevity,
            'balance': self._calculate_balance(volatility, intensity, longevity)
        }

    def _calculate_balance(self, volatility: float, intensity: float, longevity: float) -> float:
        """향 밸런스 점수 계산"""
        # 이상적인 비율과의 차이 계산
        ideal_ratio = np.array([7, 6, 7])  # 이상적인 휘발성, 강도, 지속성 비율
        actual_ratio = np.array([volatility, intensity, longevity])
        
        # 유클리드 거리 기반 밸런스 점수 계산
        distance = np.linalg.norm(ideal_ratio - actual_ratio)
        max_distance = np.linalg.norm(np.array([9, 9, 9]) - np.array([1, 1, 1]))
        
        # 0-1 범위로 정규화
        balance = 1 - (distance / max_distance)
        return balance

    def suggest_improvements(self, notes: List[str]) -> List[Dict]:
        """향 조합 개선 제안"""
        result = self.check_compatibility(notes)
        if result['score'] < 0.7:  # 호환성이 낮은 경우
            suggestions = []
            note_families = self._get_note_families(notes)

            for i, (note, family) in enumerate(zip(notes, note_families)):
                # 더 호환성 높은 대체 향료 찾기
                better_alternatives = self._find_better_alternatives(note, family, note_families)
                if better_alternatives:
                    suggestions.append({
                        'original_note': note,
                        'alternatives': better_alternatives,
                        'reason': '호환성 개선'
                    })

            return suggestions
        return []

    def _find_better_alternatives(self, note: str, family: str, other_families: List[str]) -> List[str]:
        """더 나은 대체 향료 찾기"""
        alternatives = []
        for alternative in self.scent_families[family]:
            if alternative != note:
                # 다른 향료들과의 평균 호환성 계산
                compatibility_scores = []
                for other_family in other_families:
                    if other_family != family:
                        score = self.family_compatibility[family][other_family]
                        compatibility_scores.append(score)
                
                avg_compatibility = np.mean(compatibility_scores) if compatibility_scores else 0
                if avg_compatibility > 0.7:  # 높은 호환성 기준
                    alternatives.append(alternative)
        
        return alternatives[:3]  # 상위 3개 추천 