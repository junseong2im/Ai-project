#!/usr/bin/env python3
"""
영화 장면 → 향료 레시피 생성 시스템
농도조절과 휘발성 제어에 특화
"""

from typing import Dict, List, Any
import json

class SceneFragranceRecipe:
    """장면별 향료 레시피 생성기"""
    
    def __init__(self):
        # 향료 데이터베이스 (노트별 분류)
        self.fragrance_materials = {
            # 탑노트 (Top Notes) - 첫 인상, 10분 내 사라짐
            "top_notes": {
                "lemon_oil": {"base_concentration": 5, "volatility": 95, "note_type": "citrus"},
                "peppermint": {"base_concentration": 3, "volatility": 92, "note_type": "herbal"},
                "eucalyptus": {"base_concentration": 4, "volatility": 90, "note_type": "herbal"},
                "orange_oil": {"base_concentration": 6, "volatility": 88, "note_type": "citrus"},
                "bergamot": {"base_concentration": 5, "volatility": 85, "note_type": "citrus"},
                "grapefruit": {"base_concentration": 4, "volatility": 90, "note_type": "citrus"},
                "lime": {"base_concentration": 3, "volatility": 93, "note_type": "citrus"},
                "basil": {"base_concentration": 2, "volatility": 88, "note_type": "herbal"},
                "rosemary": {"base_concentration": 3, "volatility": 85, "note_type": "herbal"}
            },
            
            # 미들노트 (Middle Notes) - 핵심 향, 30분-2시간 지속
            "middle_notes": {
                "lavender": {"base_concentration": 8, "volatility": 65, "note_type": "floral"},
                "rose_oil": {"base_concentration": 10, "volatility": 60, "note_type": "floral"},
                "jasmine": {"base_concentration": 12, "volatility": 58, "note_type": "floral"},
                "geranium": {"base_concentration": 7, "volatility": 62, "note_type": "floral"},
                "pine_needle": {"base_concentration": 6, "volatility": 68, "note_type": "woody"},
                "cinnamon": {"base_concentration": 4, "volatility": 55, "note_type": "spicy"},
                "cardamom": {"base_concentration": 3, "volatility": 58, "note_type": "spicy"},
                "black_pepper": {"base_concentration": 2, "volatility": 70, "note_type": "spicy"},
                "ylang_ylang": {"base_concentration": 8, "volatility": 52, "note_type": "floral"}
            },
            
            # 베이스노트 (Base Notes) - 깊이감, 2-8시간 지속
            "base_notes": {
                "sandalwood": {"base_concentration": 15, "volatility": 35, "note_type": "woody"},
                "cedar": {"base_concentration": 12, "volatility": 40, "note_type": "woody"},
                "vanilla": {"base_concentration": 10, "volatility": 30, "note_type": "gourmand"},
                "musk": {"base_concentration": 8, "volatility": 25, "note_type": "animalic"},
                "amber": {"base_concentration": 14, "volatility": 38, "note_type": "resinous"},
                "patchouli": {"base_concentration": 6, "volatility": 28, "note_type": "earthy"},
                "oakmoss": {"base_concentration": 5, "volatility": 32, "note_type": "earthy"},
                "benzoin": {"base_concentration": 8, "volatility": 35, "note_type": "resinous"},
                "tonka_bean": {"base_concentration": 7, "volatility": 33, "note_type": "gourmand"}
            }
        }
        
        # 장면별 휘발성 요구사항
        self.scene_volatility_map = {
            # 즉시 강한 임팩트가 필요한 장면들
            "high_volatility_scenes": [
                "action", "explosion", "chase", "fight", "surprise", "shock",
                "액션", "폭발", "추격", "싸움", "놀람", "충격"
            ],
            
            # 보통 휘발성 (일반적인 장면)
            "medium_volatility_scenes": [
                "conversation", "walking", "eating", "working", "driving",
                "대화", "걷기", "식사", "업무", "운전"
            ],
            
            # 은은한 휘발성 (조용한 장면)
            "low_volatility_scenes": [
                "sleeping", "meditation", "reading", "romantic", "peaceful", "sad",
                "잠", "명상", "독서", "로맨틱", "평화", "슬픔"
            ]
        }

    def analyze_scene_volatility(self, scene_description: str) -> str:
        """장면 설명으로부터 필요한 휘발성 레벨 결정"""
        scene_lower = scene_description.lower()
        
        # 고휘발성 키워드 확인
        for keyword in self.scene_volatility_map["high_volatility_scenes"]:
            if keyword in scene_lower:
                return "high_volatility"
        
        # 저휘발성 키워드 확인  
        for keyword in self.scene_volatility_map["low_volatility_scenes"]:
            if keyword in scene_lower:
                return "low_volatility"
        
        # 기본값은 중휘발성
        return "medium_volatility"

    def extract_scene_emotions(self, scene_description: str) -> List[str]:
        """장면에서 감정 키워드 추출"""
        scene_lower = scene_description.lower()
        emotions = []
        
        emotion_keywords = {
            "happy": ["happy", "joy", "celebration", "bright", "기쁨", "행복", "축하"],
            "romantic": ["love", "kiss", "romantic", "couple", "사랑", "키스", "로맨틱"],
            "sad": ["sad", "cry", "goodbye", "melancholy", "슬픔", "눈물", "이별"],
            "scary": ["fear", "horror", "dark", "scary", "공포", "무서운", "어둠"],
            "fresh": ["morning", "ocean", "breeze", "clean", "아침", "바다", "바람"],
            "warm": ["cozy", "home", "fireplace", "comfort", "따뜻한", "집", "편안한"],
            "mysterious": ["mystery", "secret", "hidden", "미스터리", "비밀", "숨겨진"]
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in scene_lower for keyword in keywords):
                emotions.append(emotion)
        
        return emotions if emotions else ["neutral"]

    def generate_recipe(self, scene_description: str) -> Dict[str, Any]:
        """장면 설명으로부터 향료 레시피 생성"""
        
        # 1. 휘발성 레벨 결정
        volatility_level = self.analyze_scene_volatility(scene_description)
        
        # 2. 감정 분석
        emotions = self.extract_scene_emotions(scene_description)
        
        # 3. 노트별 향료 선택 (탑/미들/베이스)
        selected_notes = self._select_notes_by_emotion_and_volatility(
            emotions[0], volatility_level
        )
        
        # 4. 노트별 농도 계산 및 레시피 구성
        recipe = {
            "scene_description": scene_description,
            "volatility_level": volatility_level,
            "detected_emotions": emotions,
            "fragrance_notes": {
                "top_notes": [],
                "middle_notes": [], 
                "base_notes": []
            },
            "mixing_instructions": []
        }
        
        total_concentration = 0
        
        # 탑노트 추가 (15-25%)
        for note_info in selected_notes["top_notes"]:
            concentration = self._calculate_concentration(
                note_info, volatility_level, "top"
            )
            recipe["fragrance_notes"]["top_notes"].append({
                "name": note_info["name"],
                "concentration_percent": concentration,
                "note_type": note_info["note_type"],
                "volatility": note_info["volatility"],
                "function": "첫인상, 즉시효과"
            })
            total_concentration += concentration
        
        # 미들노트 추가 (30-50%) 
        for note_info in selected_notes["middle_notes"]:
            concentration = self._calculate_concentration(
                note_info, volatility_level, "middle"
            )
            recipe["fragrance_notes"]["middle_notes"].append({
                "name": note_info["name"],
                "concentration_percent": concentration,
                "note_type": note_info["note_type"],
                "volatility": note_info["volatility"],
                "function": "핵심향, 메인캐릭터"
            })
            total_concentration += concentration
            
        # 베이스노트 추가 (20-35%)
        for note_info in selected_notes["base_notes"]:
            concentration = self._calculate_concentration(
                note_info, volatility_level, "base"
            )
            recipe["fragrance_notes"]["base_notes"].append({
                "name": note_info["name"],
                "concentration_percent": concentration,
                "note_type": note_info["note_type"],
                "volatility": note_info["volatility"],
                "function": "깊이감, 안정감"
            })
            total_concentration += concentration
        
        # 에탄올 베이스 (나머지 농도)
        carrier_concentration = 100 - total_concentration
        recipe["ethanol_base"] = {
            "concentration_percent": carrier_concentration,
            "function": "캐리어 솔벤트",
            "notes": "99% 에탄올 - 향료 용해 및 휘발성 조절"
        }
        
        # 5. 제조 순서 지시사항 생성
        recipe["mixing_instructions"] = self._generate_mixing_instructions(
            recipe["fragrance_notes"], volatility_level
        )
        
        # 6. 최종 레시피 정보
        recipe["total_fragrance_concentration"] = total_concentration
        recipe["duration_estimate"] = self._estimate_duration_from_notes(recipe["fragrance_notes"])
        recipe["diffusion_range"] = self._estimate_diffusion(volatility_level)
        
        return recipe

    def _select_notes_by_emotion_and_volatility(self, primary_emotion: str, volatility_level: str) -> Dict:
        """감정과 휘발성에 따른 탑/미들/베이스 노트 선택"""
        
        # 감정별 선호 향료
        emotion_preferences = {
            "romantic": {
                "top": ["bergamot", "orange_oil"],
                "middle": ["rose_oil", "jasmine", "ylang_ylang"], 
                "base": ["sandalwood", "vanilla", "musk"]
            },
            "fresh": {
                "top": ["lemon_oil", "lime", "eucalyptus"],
                "middle": ["lavender", "geranium"],
                "base": ["cedar", "oakmoss"]
            },
            "scary": {
                "top": ["basil", "rosemary"],
                "middle": ["pine_needle", "black_pepper"],
                "base": ["patchouli", "oakmoss", "cedar"]
            },
            "warm": {
                "top": ["orange_oil", "bergamot"],
                "middle": ["cinnamon", "cardamom"],
                "base": ["vanilla", "amber", "tonka_bean"]
            },
            "happy": {
                "top": ["lemon_oil", "grapefruit", "orange_oil"],
                "middle": ["lavender", "geranium"],
                "base": ["cedar", "benzoin"]
            },
            "sad": {
                "top": ["eucalyptus"],
                "middle": ["lavender", "ylang_ylang"],
                "base": ["sandalwood", "vanilla", "amber"]
            }
        }
        
        # 기본 선택
        preferences = emotion_preferences.get(primary_emotion, {
            "top": ["lemon_oil", "bergamot"], 
            "middle": ["lavender", "geranium"],
            "base": ["cedar", "sandalwood"]
        })
        
        selected = {
            "top_notes": [],
            "middle_notes": [],
            "base_notes": []
        }
        
        # 휘발성 레벨에 따른 노트 개수 조정
        if volatility_level == "high_volatility":
            # 고휘발성: 탑노트 많이, 베이스노트 적게
            top_count, middle_count, base_count = 2, 1, 1
        elif volatility_level == "low_volatility":
            # 저휘발성: 베이스노트 많이, 탑노트 적게
            top_count, middle_count, base_count = 1, 2, 2
        else:
            # 중간: 균형있게
            top_count, middle_count, base_count = 2, 2, 1
        
        # 탑노트 선택
        for i in range(min(top_count, len(preferences["top"]))):
            material_name = preferences["top"][i]
            if material_name in self.fragrance_materials["top_notes"]:
                selected["top_notes"].append({
                    "name": material_name,
                    **self.fragrance_materials["top_notes"][material_name]
                })
        
        # 미들노트 선택
        for i in range(min(middle_count, len(preferences["middle"]))):
            material_name = preferences["middle"][i]
            if material_name in self.fragrance_materials["middle_notes"]:
                selected["middle_notes"].append({
                    "name": material_name,
                    **self.fragrance_materials["middle_notes"][material_name]
                })
        
        # 베이스노트 선택
        for i in range(min(base_count, len(preferences["base"]))):
            material_name = preferences["base"][i]
            if material_name in self.fragrance_materials["base_notes"]:
                selected["base_notes"].append({
                    "name": material_name,
                    **self.fragrance_materials["base_notes"][material_name]
                })
        
        return selected

    def _generate_mixing_instructions(self, fragrance_notes: Dict, volatility_level: str) -> List[str]:
        """제조 순서 지시사항 생성"""
        
        instructions = [
            "【제조 순서 - 반드시 이 순서대로 진행하세요】",
            "",
            "1단계: 베이스노트 준비"
        ]
        
        # 베이스노트 혼합 지시
        if fragrance_notes["base_notes"]:
            instructions.append("   먼저 베이스노트들을 에탄올에 용해:")
            for note in fragrance_notes["base_notes"]:
                instructions.append(f"   • {note['name'].replace('_', ' ').title()}: {note['concentration_percent']}%")
            instructions.append("   • 10분간 저속 교반 (300rpm)")
            instructions.append("   • 30분간 정치하여 완전 용해 확인")
        
        instructions.extend([
            "",
            "2단계: 미들노트 추가"
        ])
        
        # 미들노트 혼합 지시
        if fragrance_notes["middle_notes"]:
            instructions.append("   베이스 용액에 미들노트 천천히 추가:")
            for note in fragrance_notes["middle_notes"]:
                instructions.append(f"   • {note['name'].replace('_', ' ').title()}: {note['concentration_percent']}%")
            instructions.append("   • 5분간 중속 교반 (500rpm)")
            instructions.append("   • 15분간 정치하여 향 안정화")
        
        instructions.extend([
            "",
            "3단계: 탑노트 마지막 추가"
        ])
        
        # 탑노트 혼합 지시
        if fragrance_notes["top_notes"]:
            instructions.append("   마지막에 탑노트를 가장 조심스럽게:")
            for note in fragrance_notes["top_notes"]:
                instructions.append(f"   • {note['name'].replace('_', ' ').title()}: {note['concentration_percent']}%")
            instructions.append("   • 2분간 저속 교반 (200rpm) - 탑노트 손상 방지")
            instructions.append("   • 즉시 밀폐 용기에 보관")
        
        # 휘발성별 특별 주의사항
        instructions.extend([
            "",
            "4단계: 최종 처리"
        ])
        
        if volatility_level == "high_volatility":
            instructions.extend([
                "   ⚠️ 고휘발성 제품 특별 주의사항:",
                "   • 제조 즉시 사용 (30분 내)",
                "   • 밀폐 용기 필수",
                "   • 저온 보관 (15-18°C)"
            ])
        elif volatility_level == "low_volatility":
            instructions.extend([
                "   ⚠️ 저휘발성 제품 숙성:",
                "   • 24-48시간 숙성 권장",
                "   • 암소 보관",
                "   • 사용 전 30분 실온 방치"
            ])
        else:
            instructions.extend([
                "   ⚠️ 표준 숙성 과정:",
                "   • 2-6시간 숙성",
                "   • 실온 보관 가능",
                "   • 사용 전 가볍게 흔들기"
            ])
        
        instructions.extend([
            "",
            "5단계: 품질 확인",
            "   • 색상 확인 (투명하거나 연한 황색)",
            "   • 향 밸런스 체크 (코에서 10cm 거리)",
            "   • 침전물이나 분리 현상 없는지 확인",
            "   • 최종 pH 확인 (6.0-7.5 권장)"
        ])
        
        return instructions

    def _calculate_concentration(self, material: Dict, volatility_level: str, note_position: str) -> float:
        """노트 위치와 휘발성을 고려한 농도 계산"""
        
        base_concentration = material["base_concentration"]
        
        # 노트 위치별 기본 비율
        position_ratios = {
            "top": 0.15,    # 탑노트는 적게 (15%)
            "middle": 0.35,  # 미들노트가 주력 (35%)
            "base": 0.25    # 베이스노트는 중간 (25%)
        }
        
        # 휘발성 레벨 조정
        volatility_modifiers = {
            "high_volatility": {"top": 1.5, "middle": 0.8, "base": 0.5},
            "medium_volatility": {"top": 1.0, "middle": 1.0, "base": 1.0},
            "low_volatility": {"top": 0.7, "middle": 1.2, "base": 1.4}
        }
        
        position_ratio = position_ratios.get(note_position, 0.2)
        volatility_mod = volatility_modifiers.get(volatility_level, {}).get(note_position, 1.0)
        
        final_concentration = base_concentration * position_ratio * volatility_mod
        
        # 영화용이므로 전체적으로 낮게 (최대 12%)
        return max(0.1, min(12.0, round(final_concentration, 1)))

    def _estimate_duration_from_notes(self, fragrance_notes: Dict) -> str:
        """노트 구성으로 지속시간 추정"""
        
        # 각 노트별 가중 평균 휘발성 계산
        total_volatility = 0
        total_weight = 0
        
        for note_category in fragrance_notes.values():
            for note in note_category:
                weight = note["concentration_percent"]
                total_volatility += note["volatility"] * weight
                total_weight += weight
        
        if total_weight == 0:
            return "정보 부족"
        
        avg_volatility = total_volatility / total_weight
        
        # 영화용 단시간 설정
        if avg_volatility >= 80:
            return "10-30초 (즉석 임팩트)"
        elif avg_volatility >= 65:
            return "1-2분 (단기 지속)"
        elif avg_volatility >= 45:
            return "2-5분 (중기 지속)"
        else:
            return "5-10분 (최대 지속)"


    def _estimate_diffusion(self, volatility_level: str) -> str:
        """확산 범위 추정"""
        diffusion_map = {
            "high_volatility": "1-2미터 (빠른 확산)",
            "medium_volatility": "0.5-1미터 (보통 확산)", 
            "low_volatility": "0.3미터 이내 (제한적 확산)"
        }
        return diffusion_map.get(volatility_level, "0.5-1미터")

    def format_recipe_output(self, recipe: Dict) -> str:
        """레시피를 읽기 쉬운 형태로 출력"""
        
        output = f"""
════════════════════════════════════════
🎬 영화 장면 향료 레시피
════════════════════════════════════════

📝 장면: {recipe['scene_description']}
🔥 휘발성: {recipe['volatility_level'].replace('_', ' ').title()}
😊 감정: {', '.join(recipe['detected_emotions'])}

📋 향료 구성:
"""
        
        for i, material in enumerate(recipe['materials'], 1):
            if material['function'] != 'carrier_solvent':
                output += f"""
  {i}. {material['name'].replace('_', ' ').title()}
     • 농도: {material['concentration_percent']}%
     • 기능: {material['function']}
     • 휘발성: {material['volatility']}%
     • 비고: {material['notes']}
"""
        
        # 베이스/희석제 정보
        carrier = next(m for m in recipe['materials'] if m['function'] == 'carrier_solvent')
        output += f"""
  베이스: {carrier['name'].replace('_', ' ').title()} {carrier['concentration_percent']}%
  
⏱️ 예상 지속시간: {recipe['duration_estimate']}
📍 확산 범위: {recipe['diffusion_range']}

════════════════════════════════════════
"""
        return output

def demo_recipe_generation():
    """레시피 생성 데모"""
    generator = SceneFragranceRecipe()
    
    test_scenes = [
        "액션 영화의 폭발 장면에서 강렬한 임팩트가 필요한 순간",
        "로맨틱한 해변 석양에서 커플이 키스하는 장면", 
        "조용한 도서관에서 혼자 책을 읽는 평화로운 순간",
        "공포 영화의 어두운 지하실 장면"
    ]
    
    print("🎬 영화 장면별 향료 레시피 생성 데모")
    print("=" * 50)
    
    for i, scene in enumerate(test_scenes, 1):
        print(f"\n[테스트 {i}]")
        recipe = generator.generate_recipe(scene)
        print(generator.format_recipe_output(recipe))

if __name__ == "__main__":
    demo_recipe_generation()