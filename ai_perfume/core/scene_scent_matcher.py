from __future__ import annotations

from typing import Dict, List, Optional, Any

from .scent_analyzer import ScentAnalyzer

class SceneScentMatcher:
    """장면과 향 매칭 클래스"""
    
    def __init__(self) -> None:
        self.scent_analyzer = ScentAnalyzer()
        
    def process_scene_description(
        self, 
        description: str, 
        reference_scents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """장면 설명을 처리하여 향 제조 지침 생성"""
        # 1. 향 분석
        scent_analysis = self.scent_analyzer.analyze_description(description)
        
        # 2. 레퍼런스 향 분석 (있는 경우)
        if reference_scents:
            reference_analysis = self._analyze_references(reference_scents)
            scent_analysis = self._merge_analyses(scent_analysis, reference_analysis)
        
        # 3. 원료 선정
        materials = self.scent_analyzer.get_material_recommendations(scent_analysis)
        
        # 4. 제조 지침 생성
        manufacturing_spec = self._create_manufacturing_spec(materials, scent_analysis)
        
        return {
            'analysis': scent_analysis,
            'materials': materials,
            'manufacturing': manufacturing_spec
        }
    
    def _analyze_references(self, reference_scents: List[str]) -> Dict[str, Any]:
        """레퍼런스 향수들의 특성 분석"""
        combined_analysis: Dict[str, Any] = {
            'scent_profile': [],
            'intensity': {'level': 0, 'description': ''},
            'texture': [],
            'environment': {
                'time_of_day': None,
                'season': None,
                'weather': None,
                'location': None
            }
        }
        
        # 각 레퍼런스 향 분석
        for scent in reference_scents:
            analysis = self.scent_analyzer.analyze_description(scent)
            
            # 향 프로파일 통합
            for profile in analysis['scent_profile']:
                if profile not in combined_analysis['scent_profile']:
                    combined_analysis['scent_profile'].append(profile)
            
            # 강도 평균 계산
            combined_analysis['intensity']['level'] += analysis['intensity']['level']
            
            # 질감 통합
            for texture in analysis['texture']:
                if texture not in combined_analysis['texture']:
                    combined_analysis['texture'].append(texture)
        
        # 강도 평균 계산
        if reference_scents:
            combined_analysis['intensity']['level'] /= len(reference_scents)
            combined_analysis['intensity']['description'] = self._get_intensity_description(
                combined_analysis['intensity']['level']
            )
        
        return combined_analysis
    
    def _merge_analyses(
        self, 
        scene_analysis: Dict[str, Any], 
        reference_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """장면 분석과 레퍼런스 분석 통합"""
        merged: Dict[str, Any] = {
            'scent_profile': [],
            'intensity': {'level': 0, 'description': ''},
            'texture': [],
            'environment': scene_analysis['environment']  # 환경은 장면에서만 가져옴
        }
        
        # 향 프로파일 통합 (가중치: 장면 0.7, 레퍼런스 0.3)
        scene_profiles = dict(scene_analysis['scent_profile'])
        ref_profiles = dict(reference_analysis['scent_profile'])
        
        all_categories = set(list(scene_profiles.keys()) + list(ref_profiles.keys()))
        for category in all_categories:
            score = (
                scene_profiles.get(category, 0) * 0.7 +
                ref_profiles.get(category, 0) * 0.3
            )
            if score > 0:
                merged['scent_profile'].append((category, score))
        
        # 강도 통합
        merged['intensity']['level'] = (
            scene_analysis['intensity']['level'] * 0.7 +
            reference_analysis['intensity']['level'] * 0.3
        )
        merged['intensity']['description'] = self._get_intensity_description(
            merged['intensity']['level']
        )
        
        # 질감 통합
        merged['texture'] = list(set(
            scene_analysis['texture'] + reference_analysis['texture']
        ))
        
        return merged
    
    def _get_intensity_description(self, level: float) -> str:
        """강도 수치에 따른 설명 반환"""
        if level >= 7:
            return "strong"
        elif level <= 4:
            return "weak"
        else:
            return "moderate"
    
    def _create_manufacturing_spec(
        self, 
        materials: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """제조 사양 생성"""
        # 기본 용량 설정 (100ml 기준)
        total_volume = 100
        
        # 원료 비율 계산
        material_ratios = {
            'primary': 0.5,      # 주요 원료 50%
            'secondary': 0.3,    # 보조 원료 30%
            'fixatives': 0.2     # 고정제 20%
        }
        
        # 강도에 따른 비율 조정
        concentration_factor = materials.get('concentration_factor', 1.0)
        
        # 각 원료 그룹의 구체적인 사용량 계산
        spec: Dict[str, Any] = {
            'total_volume': total_volume,
            'materials': []
        }
        
        for category in ['primary', 'secondary', 'fixatives']:
            base_ratio = material_ratios[category]
            materials_in_category = materials[category]
            
            if materials_in_category:
                # 각 원료별 비율 계산
                per_material_ratio = (base_ratio / len(materials_in_category)) * concentration_factor
                
                for material in materials_in_category:
                    spec['materials'].append({
                        'name': material,
                        'volume': round(total_volume * per_material_ratio, 2),
                        'ratio': round(per_material_ratio * 100, 1)
                    })
        
        # 제조 방법 추가
        spec['manufacturing_method'] = self._determine_manufacturing_method(analysis)
        
        return spec
    
    def _determine_manufacturing_method(self, analysis: Dict[str, Any]) -> str:
        """분석 결과에 따른 제조 방법 결정"""
        if any(texture in ['heavy', 'dry'] for texture in analysis['texture']):
            return 'dilution_blending'  # 희석 배합
        else:
            return 'direct_blending'    # 직접 배합 