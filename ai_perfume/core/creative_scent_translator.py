from typing import Dict, List, Tuple
import re

class CreativeScentTranslator:
    def __init__(self):
        # 감정과 이미지를 향 요소로 변환하는 매핑
        self.emotional_scent_mapping = {
            # 긍정적 감정
            "행복": {
                "notes": ["오렌지 블로섬", "바닐라", "일랑일랑"],
                "intensity": "moderate to strong",
                "characteristics": ["달콤한", "따뜻한", "밝은"]
            },
            "평화": {
                "notes": ["라벤더", "캐모마일", "베르가못"],
                "intensity": "soft to moderate",
                "characteristics": ["부드러운", "깨끗한", "차분한"]
            },
            # 부정적 감정
            "슬픔": {
                "notes": ["아이리스", "바이올렛", "샌달우드"],
                "intensity": "soft",
                "characteristics": ["파우더리", "우아한", "그윽한"]
            },
            "분노": {
                "notes": ["블랙페퍼", "시나몬", "파출리"],
                "intensity": "strong",
                "characteristics": ["스파이시", "강렬한", "진한"]
            }
        }

        # 자연 요소 매핑
        self.nature_scent_mapping = {
            "바다": {
                "notes": ["씨솔트", "마린노트", "알게"],
                "modifiers": ["오존", "코코넛", "베르가못"]
            },
            "숲": {
                "notes": ["파인니들", "시더우드", "베티버"],
                "modifiers": ["이끼", "허브", "페퍼민트"]
            },
            "비": {
                "notes": ["페티그레인", "베티버", "민트"],
                "modifiers": ["오존", "씨솔트", "머스크"]
            },
            "눈": {
                "notes": ["민트", "화이트머스크", "시더우드"],
                "modifiers": ["아이리스", "바닐라", "통카빈"]
            }
        }

        # 시간대별 향 특성
        self.time_scent_mapping = {
            "새벽": {
                "characteristics": ["차가운", "투명한", "깨끗한"],
                "notes": ["민트", "베르가못", "화이트티"]
            },
            "아침": {
                "characteristics": ["상쾌한", "밝은", "활기찬"],
                "notes": ["시트러스", "그린티", "네롤리"]
            },
            "낮": {
                "characteristics": ["따뜻한", "선명한", "강렬한"],
                "notes": ["오렌지블로섬", "일랑일랑", "자스민"]
            },
            "저녁": {
                "characteristics": ["부드러운", "로맨틱한", "세련된"],
                "notes": ["바이올렛", "앰버", "바닐라"]
            },
            "밤": {
                "characteristics": ["신비로운", "관능적인", "깊은"],
                "notes": ["샌달우드", "통카빈", "머스크"]
            }
        }

        # 질감 매핑
        self.texture_scent_mapping = {
            "부드러운": ["바닐라", "통카빈", "화이트머스크"],
            "거친": ["베티버", "파출리", "시더우드"],
            "차가운": ["민트", "유칼립투스", "주니퍼베리"],
            "따뜻한": ["시나몬", "통카빈", "앰버"],
            "습한": ["페티그레인", "오크모스", "아이리스"],
            "건조한": ["시더우드", "베티버", "샌달우드"]
        }

    def translate_creative_description(self, description: str) -> Dict:
        """창의적인 향 설명을 실제 향 구성요소로 변환"""
        # 기본 분석 결과 구조
        analysis = {
            'primary_notes': [],    # 주요 향료
            'supporting_notes': [], # 보조 향료
            'base_notes': [],       # 베이스 향료
            'characteristics': [],  # 향의 특성
            'intensity': 5.0,       # 기본 강도
            'texture': [],          # 질감
            'creative_elements': [] # 창의적 요소
        }

        # 감정 분석
        emotions = self._analyze_emotions(description)
        self._add_emotional_notes(analysis, emotions)

        # 자연 요소 분석
        nature_elements = self._analyze_nature_elements(description)
        self._add_nature_notes(analysis, nature_elements)

        # 시간대 분석
        time_elements = self._analyze_time_elements(description)
        self._add_time_based_notes(analysis, time_elements)

        # 질감 분석
        textures = self._analyze_textures(description)
        self._add_texture_notes(analysis, textures)

        # 창의적 요소 추출
        creative_elements = self._extract_creative_elements(description)
        analysis['creative_elements'] = creative_elements

        # 강도 조정
        analysis['intensity'] = self._calculate_intensity(emotions, nature_elements, time_elements)

        # 중복 제거 및 정리
        self._cleanup_notes(analysis)

        return analysis

    def _analyze_emotions(self, description: str) -> List[Tuple[str, float]]:
        """감정 요소 분석"""
        emotions = []
        for emotion in self.emotional_scent_mapping.keys():
            if emotion in description.lower():
                # 감정 강도 계산 (문맥에 따라 0.0-1.0)
                intensity = 0.5  # 기본값
                # 강조 표현 확인
                if re.search(f"매우 {emotion}|너무 {emotion}|강한 {emotion}", description):
                    intensity = 0.8
                elif re.search(f"약간 {emotion}|살짝 {emotion}", description):
                    intensity = 0.3
                emotions.append((emotion, intensity))
        return emotions

    def _analyze_nature_elements(self, description: str) -> List[str]:
        """자연 요소 분석"""
        elements = []
        for element in self.nature_scent_mapping.keys():
            if element in description:
                elements.append(element)
        return elements

    def _analyze_time_elements(self, description: str) -> List[str]:
        """시간 요소 분석"""
        times = []
        for time in self.time_scent_mapping.keys():
            if time in description:
                times.append(time)
        return times

    def _analyze_textures(self, description: str) -> List[str]:
        """질감 요소 분석"""
        textures = []
        for texture in self.texture_scent_mapping.keys():
            if texture in description:
                textures.append(texture)
        return textures

    def _add_emotional_notes(self, analysis: Dict, emotions: List[Tuple[str, float]]):
        """감정 기반 향료 추가"""
        for emotion, intensity in emotions:
            if emotion in self.emotional_scent_mapping:
                notes = self.emotional_scent_mapping[emotion]["notes"]
                analysis['primary_notes'].extend(notes[:1])  # 주요 향료
                analysis['supporting_notes'].extend(notes[1:2])  # 보조 향료
                analysis['base_notes'].extend(notes[2:])  # 베이스 향료
                analysis['characteristics'].extend(
                    self.emotional_scent_mapping[emotion]["characteristics"]
                )

    def _add_nature_notes(self, analysis: Dict, elements: List[str]):
        """자연 요소 기반 향료 추가"""
        for element in elements:
            if element in self.nature_scent_mapping:
                mapping = self.nature_scent_mapping[element]
                analysis['primary_notes'].extend(mapping["notes"][:1])
                analysis['supporting_notes'].extend(mapping["notes"][1:2])
                analysis['base_notes'].extend(mapping["notes"][2:])
                analysis['supporting_notes'].extend(mapping["modifiers"][:2])

    def _add_time_based_notes(self, analysis: Dict, times: List[str]):
        """시간대 기반 향료 추가"""
        for time in times:
            if time in self.time_scent_mapping:
                mapping = self.time_scent_mapping[time]
                analysis['characteristics'].extend(mapping["characteristics"])
                analysis['supporting_notes'].extend(mapping["notes"])

    def _add_texture_notes(self, analysis: Dict, textures: List[str]):
        """질감 기반 향료 추가"""
        for texture in textures:
            if texture in self.texture_scent_mapping:
                analysis['texture'].append(texture)
                analysis['supporting_notes'].extend(self.texture_scent_mapping[texture][:1])
                analysis['base_notes'].extend(self.texture_scent_mapping[texture][1:])

    def _extract_creative_elements(self, description: str) -> List[str]:
        """창의적 표현 요소 추출"""
        # 비유적 표현 추출
        metaphors = re.findall(r'마치 (.+?)같은|처럼 (.+?)[.|,]', description)
        # 감각적 표현 추출
        sensory = re.findall(r'([\w]+스러운|[\w]+한|[\w]+적인)', description)
        
        creative_elements = []
        for m in metaphors:
            creative_elements.extend([x for x in m if x])
        creative_elements.extend(sensory)
        
        return list(set(creative_elements))  # 중복 제거

    def _calculate_intensity(self, emotions: List[Tuple[str, float]], 
                           nature_elements: List[str], 
                           time_elements: List[str]) -> float:
        """향의 강도 계산"""
        base_intensity = 5.0
        
        # 감정 강도 반영
        emotion_intensity = sum(intensity for _, intensity in emotions)
        if emotion_intensity > 0:
            base_intensity += emotion_intensity

        # 자연 요소 반영
        nature_modifiers = {
            "바다": 1.0,  # 강한 향
            "숲": 0.5,   # 중간 향
            "비": -0.5,  # 약한 향
            "눈": -1.0   # 매우 약한 향
        }
        for element in nature_elements:
            if element in nature_modifiers:
                base_intensity += nature_modifiers[element]

        # 시간대 반영
        time_modifiers = {
            "새벽": -0.5,
            "아침": 0.0,
            "낮": 1.0,
            "저녁": 0.0,
            "밤": -0.5
        }
        for time in time_elements:
            if time in time_modifiers:
                base_intensity += time_modifiers[time]

        # 1-10 범위로 제한
        return max(1.0, min(10.0, base_intensity))

    def _cleanup_notes(self, analysis: Dict):
        """향료 중복 제거 및 정리"""
        # 중복 제거
        analysis['primary_notes'] = list(set(analysis['primary_notes']))
        analysis['supporting_notes'] = list(set(analysis['supporting_notes']))
        analysis['base_notes'] = list(set(analysis['base_notes']))
        analysis['characteristics'] = list(set(analysis['characteristics']))

        # 주요 향료가 다른 카테고리에 있으면 제거
        for note in analysis['primary_notes']:
            if note in analysis['supporting_notes']:
                analysis['supporting_notes'].remove(note)
            if note in analysis['base_notes']:
                analysis['base_notes'].remove(note)

        # 보조 향료가 베이스에 있으면 제거
        for note in analysis['supporting_notes']:
            if note in analysis['base_notes']:
                analysis['base_notes'].remove(note)

        # 각 카테고리 최대 개수 제한
        analysis['primary_notes'] = analysis['primary_notes'][:3]
        analysis['supporting_notes'] = analysis['supporting_notes'][:4]
        analysis['base_notes'] = analysis['base_notes'][:3] 