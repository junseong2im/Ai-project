import json
import os
from typing import List, Dict, Any
import math

class FragranceManufacturingManager:
    def __init__(self, database_path: str = "data/fragrance_materials_database.json"):
        self.database_path = database_path
        self.database = self._load_database()
        
    def _load_database(self) -> Dict:
        """원료 데이터베이스 로드"""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        raise FileNotFoundError(f"데이터베이스 파일을 찾을 수 없습니다: {self.database_path}")
    
    def get_material_info(self, material_name: str) -> Dict:
        """원료 정보 조회"""
        # 에센셜 오일 검색
        for category in self.database['raw_materials']['essential_oils'].values():
            for material in category:
                if material['name'] == material_name:
                    return material
        
        # 앱솔루트 검색
        for material in self.database['raw_materials']['absolutes']:
            if material['name'] == material_name:
                return material
        
        # 합성 원료 검색
        for material in self.database['raw_materials']['synthetic_materials']:
            if material['name'] == material_name:
                return material
                
        return None
    
    def calculate_blend(self, materials: List[Dict], total_volume: float) -> Dict:
        """블렌딩 비율 계산"""
        result = {
            'materials': [],
            'carrier': None,
            'total_volume': total_volume,
            'processing_method': 'direct_blending'
        }
        
        # 총 농도 계산
        total_concentration = sum(m['concentration'] for m in materials)
        if total_concentration > 100:
            raise ValueError("총 농도가 100%를 초과할 수 없습니다.")
        
        # 캐리어 필요 여부 확인
        carrier_needed = 100 - total_concentration
        if carrier_needed > 0:
            # 캐리어 선택 (에탄올 또는 호호바 오일)
            carrier = self.select_carrier(materials)
            result['carrier'] = {
                'name': carrier['name'],
                'volume': (carrier_needed / 100) * total_volume,
                'percentage': carrier_needed
            }
        
        # 각 원료별 부피 계산
        for material in materials:
            material_volume = (material['concentration'] / 100) * total_volume
            material_info = self.get_material_info(material['name'])
            
            result['materials'].append({
                'name': material['name'],
                'volume': material_volume,
                'percentage': material['concentration'],
                'properties': material_info['properties'] if material_info else None
            })
        
        return result
    
    def select_carrier(self, materials: List[Dict]) -> Dict:
        """적합한 캐리어 선택"""
        # 원료들의 용해성 확인
        alcohol_soluble = all(
            self.get_material_info(m['name'])['properties']['solubility']['in_alcohol'] == "우수"
            for m in materials
        )
        
        if alcohol_soluble:
            return self.database['blending_guidelines']['carrier_materials']['alcohols'][0]
        else:
            return self.database['blending_guidelines']['carrier_materials']['oils'][0]
    
    def get_manufacturing_process(self, materials: List[Dict]) -> Dict:
        """제조 공정 정보 반환"""
        # 원료 특성에 따른 공정 선택
        needs_heating = any(
            self.get_material_info(m['name'])['properties'].get('melting_point', 0) > 25
            for m in materials
        )
        
        if needs_heating:
            process = 'dilution_blending'
        else:
            process = 'direct_blending'
            
        return {
            'process': self.database['manufacturing_processes']['blending_methods'][process],
            'quality_control': self.get_quality_control_steps(materials)
        }
    
    def get_quality_control_steps(self, materials: List[Dict]) -> List[str]:
        """품질 관리 단계 반환"""
        steps = []
        
        # 기본 물성 검사
        steps.extend(self.database['quality_control']['testing_methods']['physical_tests'])
        
        # 원료별 특수 검사 항목 추가
        for material in materials:
            material_info = self.get_material_info(material['name'])
            if material_info.get('properties', {}).get('light_sensitivity') == "높음":
                steps.append("광안정성 테스트")
            if material_info.get('properties', {}).get('volatility_rate', 0) > 7:
                steps.append("휘발성 테스트")
                
        return list(set(steps))  # 중복 제거
    
    def generate_manufacturing_instructions(self, blend_info: Dict) -> Dict:
        """제조 지침 생성"""
        process = self.get_manufacturing_process(blend_info['materials'])
        
        instructions = {
            'preparation': {
                'equipment': process['process']['equipment_needed'],
                'materials': [
                    {
                        'name': m['name'],
                        'amount': f"{m['volume']:.2f}ml ({m['percentage']}%)"
                    }
                    for m in blend_info['materials']
                ]
            },
            'steps': [],
            'quality_control': process['quality_control']
        }
        
        # 캐리어가 있는 경우 추가
        if blend_info['carrier']:
            instructions['preparation']['materials'].append({
                'name': blend_info['carrier']['name'],
                'amount': f"{blend_info['carrier']['volume']:.2f}ml ({blend_info['carrier']['percentage']}%)"
            })
        
        # 제조 단계 생성
        instructions['steps'] = self._generate_detailed_steps(blend_info, process['process']['steps'])
        
        return instructions
    
    def _generate_detailed_steps(self, blend_info: Dict, base_steps: List[str]) -> List[Dict]:
        """상세 제조 단계 생성"""
        detailed_steps = []
        
        for i, step in enumerate(base_steps, 1):
            if step == "원료 계량":
                for material in blend_info['materials']:
                    detailed_steps.append({
                        'step': f"{i}.{len(detailed_steps)+1}",
                        'description': f"{material['name']} {material['volume']:.2f}ml를 정밀 계량합니다.",
                        'caution': "정확한 계량이 중요합니다."
                    })
            elif step == "캐리어 준비" and blend_info.get('carrier'):
                detailed_steps.append({
                    'step': f"{i}.1",
                    'description': f"{blend_info['carrier']['name']} {blend_info['carrier']['volume']:.2f}ml를 준비합니다.",
                    'caution': "캐리어는 실온에서 준비합니다."
                })
            elif step == "순차적 혼합":
                detailed_steps.append({
                    'step': f"{i}.1",
                    'description': "휘발성이 높은 원료부터 순차적으로 혼합합니다.",
                    'caution': "천천히 혼합하여 기포가 생기지 않도록 합니다."
                })
            elif step == "숙성":
                detailed_steps.append({
                    'step': f"{i}.1",
                    'description': "혼합물을 밀봉하여 어둡고 서늘한 곳에서 1-2주간 숙성시킵니다.",
                    'caution': "숙성 중 주기적으로 향을 체크합니다."
                })
                
        return detailed_steps 