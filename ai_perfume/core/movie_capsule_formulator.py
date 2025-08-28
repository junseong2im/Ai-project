#!/usr/bin/env python3
"""
영화 촬영용 캡슐 방향제 제조 시스템
감독 요구사항: 장면 → 향 → 3-10초 지속 → 낮은 확산성 → 정확한 제조 공식
"""

import math
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from core.deep_learning_integration import get_trained_predictor

@dataclass
class CapsuleFormula:
    """캡슐 방향제 공식"""
    scene_description: str
    target_duration: float  # 3-10초
    diffusion_control: float  # 낮은 확산성
    raw_materials: List[Dict]
    mixing_ratios: Dict[str, float]
    production_sequence: List[str]
    encapsulation_method: str
    activation_mechanism: str
    estimated_cost_per_unit: float

class MovieCapsuleFormulator:
    """영화용 캡슐 방향제 제조 시스템"""
    
    def __init__(self):
        # 캡슐용 특별 원료 (빠른 휘발 + 제어된 확산)
        self.capsule_materials = {
            # 초고휘발성 원료 (1-3초 지속)
            "burst_notes": {
                "peppermint_oil": {"volatility": 0.95, "intensity": 90, "cost_per_g": 0.15},
                "eucalyptus": {"volatility": 0.92, "intensity": 85, "cost_per_g": 0.12},
                "lemon_aldehyde": {"volatility": 0.88, "intensity": 80, "cost_per_g": 0.18},
                "spearmint": {"volatility": 0.90, "intensity": 75, "cost_per_g": 0.14}
            },
            
            # 단기지속 원료 (3-7초)
            "short_notes": {
                "bergamot": {"volatility": 0.75, "intensity": 70, "cost_per_g": 0.25},
                "orange_terpenes": {"volatility": 0.72, "intensity": 65, "cost_per_g": 0.20},
                "lavender_head": {"volatility": 0.68, "intensity": 60, "cost_per_g": 0.22},
                "pine_needles": {"volatility": 0.70, "intensity": 68, "cost_per_g": 0.19},
                "rosemary": {"volatility": 0.73, "intensity": 72, "cost_per_g": 0.16}
            },
            
            # 중간지속 원료 (5-10초)  
            "medium_notes": {
                "geranium": {"volatility": 0.55, "intensity": 50, "cost_per_g": 0.28},
                "rose_petals": {"volatility": 0.52, "intensity": 48, "cost_per_g": 0.35},
                "jasmine_light": {"volatility": 0.58, "intensity": 52, "cost_per_g": 0.40},
                "cedar_tips": {"volatility": 0.50, "intensity": 45, "cost_per_g": 0.24},
                "sandalwood_light": {"volatility": 0.48, "intensity": 42, "cost_per_g": 0.45}
            },
            
            # 확산 억제제 (향이 멀리 퍼지지 않도록)
            "diffusion_controllers": {
                "glycerin": {"density": 1.26, "viscosity_modifier": 0.8, "cost_per_g": 0.03},
                "propylene_glycol": {"density": 1.04, "viscosity_modifier": 0.6, "cost_per_g": 0.04},
                "dipropylene_glycol": {"density": 1.02, "viscosity_modifier": 0.7, "cost_per_g": 0.05}
            },
            
            # 캡슐 외피 재료
            "capsule_shells": {
                "gelatin_type_A": {"thickness": 0.1, "burst_pressure": 2.5, "cost_per_unit": 0.008},
                "hydroxypropyl_starch": {"thickness": 0.08, "burst_pressure": 1.8, "cost_per_unit": 0.006},
                "chitosan_blend": {"thickness": 0.12, "burst_pressure": 3.2, "cost_per_unit": 0.012}
            }
        }
        
        # 장면별 향기 매핑
        self.scene_fragrance_map = {
            "romantic": {"primary": "rose_petals", "secondary": "jasmine_light", "accent": "bergamot"},
            "mystery": {"primary": "cedar_tips", "secondary": "pine_needles", "accent": "eucalyptus"},
            "happy": {"primary": "orange_terpenes", "secondary": "lemon_aldehyde", "accent": "spearmint"},
            "sad": {"primary": "lavender_head", "secondary": "geranium", "accent": "sandalwood_light"},
            "action": {"primary": "peppermint_oil", "secondary": "rosemary", "accent": "pine_needles"},
            "peaceful": {"primary": "lavender_head", "secondary": "sandalwood_light", "accent": "bergamot"},
            "scary": {"primary": "cedar_tips", "secondary": "eucalyptus", "accent": "pine_needles"},
            "nostalgic": {"primary": "rose_petals", "secondary": "sandalwood_light", "accent": "lavender_head"}
        }
        
        # 딥러닝 모델 로드
        try:
            self.trained_predictor = get_trained_predictor()
            self.ml_available = self.trained_predictor.is_loaded
        except:
            self.trained_predictor = None
            self.ml_available = False
    
    def analyze_scene_for_scent(self, scene_description: str) -> Dict:
        """영화 장면을 분석하여 향기 특성 결정"""
        scene_lower = scene_description.lower()
        
        # 감정 키워드 감지
        emotions = []
        emotion_keywords = {
            "romantic": ["love", "kiss", "romance", "romantic", "couple", "heart", "사랑", "키스", "로맨틱"],
            "mystery": ["dark", "mysterious", "secret", "shadow", "어둠", "미스터리", "비밀"],
            "happy": ["happy", "joy", "bright", "sunny", "celebration", "기쁨", "행복", "밝은"],
            "sad": ["sad", "cry", "tears", "goodbye", "melancholy", "슬픈", "눈물", "이별"],
            "action": ["fight", "chase", "explosion", "fast", "액션", "싸움", "추격"],
            "peaceful": ["calm", "serene", "quiet", "meditation", "평화", "고요", "명상"],
            "scary": ["horror", "fear", "scary", "ghost", "공포", "무서운", "귀신"],
            "nostalgic": ["memory", "past", "childhood", "old", "추억", "과거", "어린시절"]
        }
        
        detected_emotions = []
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in scene_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        # 기본 감정 설정
        if not detected_emotions:
            detected_emotions = ["peaceful"]
        
        # 시간대/환경 분석
        time_modifiers = {
            "morning": 1.2,
            "afternoon": 1.0, 
            "evening": 0.8,
            "night": 0.6
        }
        
        environment_modifiers = {
            "outdoor": 1.3,
            "indoor": 0.9,
            "beach": 1.1,
            "forest": 1.0,
            "city": 0.7
        }
        
        time_factor = 1.0
        env_factor = 1.0
        
        for time_key, factor in time_modifiers.items():
            if time_key in scene_lower:
                time_factor = factor
                break
        
        for env_key, factor in environment_modifiers.items():
            if env_key in scene_lower:
                env_factor = factor
                break
        
        return {
            "detected_emotions": detected_emotions,
            "primary_emotion": detected_emotions[0],
            "time_factor": time_factor,
            "environment_factor": env_factor,
            "intensity_modifier": time_factor * env_factor,
            "scene_complexity": len(scene_description.split()) / 20.0
        }
    
    def formulate_capsule(self, scene_description: str, target_duration: float = 7.0) -> CapsuleFormula:
        """영화 장면에 맞는 캡슐 방향제 공식 생성"""
        
        # 1. 장면 분석
        scene_analysis = self.analyze_scene_for_scent(scene_description)
        primary_emotion = scene_analysis["primary_emotion"]
        intensity_mod = scene_analysis["intensity_modifier"]
        
        # 2. 딥러닝 모델 예측 (가능한 경우)
        ml_predictions = None
        if self.ml_available:
            try:
                ml_result = self.trained_predictor.predict_scene_fragrance(scene_description)
                if ml_result["success"]:
                    ml_predictions = ml_result["predictions"]
            except:
                pass
        
        # 3. 기본 향료 선택
        fragrance_profile = self.scene_fragrance_map.get(primary_emotion, self.scene_fragrance_map["peaceful"])
        
        # 4. 지속시간별 원료 분배
        raw_materials = []
        mixing_ratios = {}
        
        # 지속시간에 따른 원료 비율 계산
        if target_duration <= 3:
            # 초단기 (1-3초): 거의 burst만
            burst_ratio = 0.70
            short_ratio = 0.25
            medium_ratio = 0.05
        elif target_duration <= 6:
            # 단기 (3-6초): burst + short 중심
            burst_ratio = 0.40
            short_ratio = 0.50
            medium_ratio = 0.10
        else:
            # 중기 (6-10초): 균형잡힌 배합
            burst_ratio = 0.25
            short_ratio = 0.45
            medium_ratio = 0.30
        
        # 5. 주원료 선택 및 비율 계산
        total_fragrance_volume = 0.8  # 80%는 향료, 20%는 베이스/확산억제제
        
        # Primary 향료 (40%)
        primary_name = fragrance_profile["primary"]
        primary_category = self._find_material_category(primary_name)
        primary_amount = total_fragrance_volume * 0.40
        
        raw_materials.append({
            "name": primary_name,
            "category": primary_category,
            "amount_ml": round(primary_amount, 3),
            "percentage": 40.0,
            "function": "primary_scent",
            "properties": self.capsule_materials[primary_category][primary_name]
        })
        mixing_ratios[primary_name] = primary_amount
        
        # Secondary 향료 (30%)
        secondary_name = fragrance_profile["secondary"] 
        secondary_category = self._find_material_category(secondary_name)
        secondary_amount = total_fragrance_volume * 0.30
        
        raw_materials.append({
            "name": secondary_name,
            "category": secondary_category,
            "amount_ml": round(secondary_amount, 3),
            "percentage": 30.0,
            "function": "secondary_scent",
            "properties": self.capsule_materials[secondary_category][secondary_name]
        })
        mixing_ratios[secondary_name] = secondary_amount
        
        # Accent 향료 (10%)
        accent_name = fragrance_profile["accent"]
        accent_category = self._find_material_category(accent_name)
        accent_amount = total_fragrance_volume * 0.10
        
        raw_materials.append({
            "name": accent_name,
            "category": accent_category, 
            "amount_ml": round(accent_amount, 3),
            "percentage": 10.0,
            "function": "accent_scent",
            "properties": self.capsule_materials[accent_category][accent_name]
        })
        mixing_ratios[accent_name] = accent_amount
        
        # 6. 확산 억제제 (15%)
        diffusion_controller = "propylene_glycol"
        controller_amount = 0.15
        
        raw_materials.append({
            "name": diffusion_controller,
            "category": "diffusion_controllers",
            "amount_ml": round(controller_amount, 3),
            "percentage": 15.0,
            "function": "diffusion_control",
            "properties": self.capsule_materials["diffusion_controllers"][diffusion_controller]
        })
        mixing_ratios[diffusion_controller] = controller_amount
        
        # 7. 베이스 오일 (5%)
        base_oil_amount = 0.05
        raw_materials.append({
            "name": "neutral_carrier_oil",
            "category": "base",
            "amount_ml": round(base_oil_amount, 3),
            "percentage": 5.0,
            "function": "carrier_base",
            "properties": {"density": 0.92, "cost_per_g": 0.02}
        })
        mixing_ratios["neutral_carrier_oil"] = base_oil_amount
        
        # 8. 제조 순서
        production_sequence = [
            "1단계: 베이스 오일 준비 (상온, 25°C)",
            "2단계: 확산 억제제(프로필렌글리콜) 베이스와 혼합",
            "3단계: Primary 향료 천천히 투입하며 저속 교반 (200rpm, 2분)",
            "4단계: Secondary 향료 추가 후 중속 교반 (400rpm, 1분)",
            "5단계: Accent 향료 마지막 투입 (고속 교반 800rpm, 30초)",
            "6단계: 혼합물을 30분간 정치 (기포 제거)",
            "7단계: 0.1ml씩 젤라틴 캡슐에 충진",
            "8단계: 캡슐 밀봉 및 24시간 안정화",
            "9단계: 품질 검사 (향 강도, 지속시간, 확산성 테스트)"
        ]
        
        # 9. 캡슐 사양
        capsule_spec = self.capsule_materials["capsule_shells"]["gelatin_type_A"]
        encapsulation_method = f"젤라틴 타입A 캡슐 (두께: {capsule_spec['thickness']}mm, 파열압: {capsule_spec['burst_pressure']}kPa)"
        activation_mechanism = "수동 압박시 파열 (손가락 압력 2-3kPa로 터짐)"
        
        # 10. 비용 계산
        total_cost = 0.0
        for material in raw_materials:
            if "cost_per_g" in material["properties"]:
                # 밀도 고려하여 ml -> g 변환 (대부분 향료는 밀도 ~0.9)
                density = material["properties"].get("density", 0.9)
                weight_g = material["amount_ml"] * density
                material_cost = weight_g * material["properties"]["cost_per_g"]
                total_cost += material_cost
        
        # 캡슐 비용 추가
        total_cost += capsule_spec["cost_per_unit"]
        
        # ML 예측으로 수정사항 적용
        if ml_predictions:
            # 예측된 지속시간과 목표가 다르면 비율 조정
            predicted_duration = ml_predictions.get("longevity_hours", 0) * 3600  # 초로 변환
            if abs(predicted_duration - target_duration) > 2:
                duration_adjustment = target_duration / max(predicted_duration, 1)
                # 휘발성 높은 원료 비율 조정
                for material in raw_materials:
                    if material["category"] in ["burst_notes", "short_notes"]:
                        material["amount_ml"] *= duration_adjustment
                        mixing_ratios[material["name"]] *= duration_adjustment
        
        return CapsuleFormula(
            scene_description=scene_description,
            target_duration=target_duration,
            diffusion_control=0.3,  # 낮은 확산성 (0-1 스케일)
            raw_materials=raw_materials,
            mixing_ratios=mixing_ratios,
            production_sequence=production_sequence,
            encapsulation_method=encapsulation_method,
            activation_mechanism=activation_mechanism,
            estimated_cost_per_unit=round(total_cost, 4)
        )
    
    def _find_material_category(self, material_name: str) -> str:
        """원료명으로 카테고리 찾기"""
        for category, materials in self.capsule_materials.items():
            if material_name in materials:
                return category
        return "medium_notes"  # 기본값
    
    def generate_detailed_report(self, formula: CapsuleFormula) -> str:
        """상세 제조 보고서 생성"""
        report = f"""
═══════════════════════════════════════════════════════════════
🎬 영화용 캡슐 방향제 제조 명세서
═══════════════════════════════════════════════════════════════

📝 장면 설명: {formula.scene_description}
⏱️ 목표 지속시간: {formula.target_duration}초
📍 확산 범위: {formula.diffusion_control * 100:.1f}cm 반경 (낮은 확산)

🧪 원료 구성 (총 1.0ml 기준):
"""
        
        for i, material in enumerate(formula.raw_materials, 1):
            props = material["properties"]
            cost_per_unit = material["amount_ml"] * props.get("cost_per_g", 0) * props.get("density", 0.9)
            
            report += f"""
  {i}. {material["name"].replace('_', ' ').title()}
     • 용량: {material["amount_ml"]:.3f}ml ({material["percentage"]:.1f}%)
     • 기능: {material["function"]}
     • 특성: {props}
     • 단가: ${cost_per_unit:.4f}
"""
        
        report += f"""
💰 예상 제조 비용: ${formula.estimated_cost_per_unit:.4f}/개

🏭 제조 공정:
"""
        for step in formula.production_sequence:
            report += f"   {step}\n"
            
        report += f"""

📦 캡슐 사양:
   • 포장 방식: {formula.encapsulation_method}
   • 활성화: {formula.activation_mechanism}
   • 1회 용량: 0.1ml (10회 터뜨리기 가능)

⚠️  주의사항:
   • 촬영 직전에 터뜨려야 최적 효과
   • 실내 촬영시 환기 필수
   • 배우 알레르기 사전 확인 권장
   • 25°C 이하 서늘한 곳 보관

═══════════════════════════════════════════════════════════════
"""
        
        return report

# 전역 인스턴스
_capsule_formulator = None

def get_capsule_formulator():
    """캡슐 제조기 싱글톤 인스턴스"""
    global _capsule_formulator
    if _capsule_formulator is None:
        _capsule_formulator = MovieCapsuleFormulator()
    return _capsule_formulator