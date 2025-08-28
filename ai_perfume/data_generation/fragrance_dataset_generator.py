#!/usr/bin/env python3
"""
대규모 향료 원료 데이터셋 생성기
200,000개 자연/합성 향료 데이터 생성 (편향 없는 딥러닝용)
"""

import json
import random
import numpy as np
from typing import Dict, List, Tuple
import csv
import os
from pathlib import Path

class FragranceDatasetGenerator:
    """향료 원료 대규모 데이터셋 생성기"""
    
    def __init__(self):
        self.output_dir = Path("../data/datasets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 자연 유래 원료 기본 데이터
        self.natural_materials = self._initialize_natural_materials()
        
        # 합성 원료 기본 데이터
        self.synthetic_materials = self._initialize_synthetic_materials()
        
        # 향기 계열 분류
        self.scent_families = {
            "citrus": {"weight": 0.12, "natural_ratio": 0.85},
            "floral": {"weight": 0.15, "natural_ratio": 0.75},
            "woody": {"weight": 0.13, "natural_ratio": 0.80},
            "oriental": {"weight": 0.10, "natural_ratio": 0.60},
            "fresh": {"weight": 0.09, "natural_ratio": 0.70},
            "fruity": {"weight": 0.08, "natural_ratio": 0.65},
            "green": {"weight": 0.07, "natural_ratio": 0.90},
            "aquatic": {"weight": 0.06, "natural_ratio": 0.20},
            "spicy": {"weight": 0.06, "natural_ratio": 0.75},
            "gourmand": {"weight": 0.05, "natural_ratio": 0.45},
            "herbal": {"weight": 0.04, "natural_ratio": 0.95},
            "mineral": {"weight": 0.03, "natural_ratio": 0.10},
            "smoky": {"weight": 0.02, "natural_ratio": 0.30}
        }
        
        # 추출 방법별 데이터
        self.extraction_methods = {
            "steam_distillation": {"cost_factor": 1.0, "purity": 0.85, "natural_only": True},
            "cold_press": {"cost_factor": 0.8, "purity": 0.90, "natural_only": True},
            "solvent_extraction": {"cost_factor": 1.2, "purity": 0.75, "natural_only": True},
            "co2_extraction": {"cost_factor": 2.0, "purity": 0.95, "natural_only": True},
            "enfluerage": {"cost_factor": 5.0, "purity": 0.60, "natural_only": True},
            "chemical_synthesis": {"cost_factor": 0.3, "purity": 0.99, "natural_only": False},
            "biotechnology": {"cost_factor": 1.5, "purity": 0.92, "natural_only": False}
        }
        
    def _initialize_natural_materials(self) -> Dict:
        """자연 유래 원료 기본 템플릿"""
        return {
            # 감귤류
            "citrus_oils": [
                {"base_name": "bergamot", "variations": ["calabrian", "turkish"], "intensity_range": (7, 9)},
                {"base_name": "lemon", "variations": ["sicilian", "meyer", "eureka"], "intensity_range": (8, 10)},
                {"base_name": "lime", "variations": ["persian", "kaffir", "mexican"], "intensity_range": (8, 9)},
                {"base_name": "orange", "variations": ["sweet", "bitter", "blood"], "intensity_range": (6, 8)},
                {"base_name": "grapefruit", "variations": ["pink", "white", "ruby"], "intensity_range": (7, 9)},
                {"base_name": "mandarin", "variations": ["green", "red", "clementine"], "intensity_range": (5, 7)},
                {"base_name": "yuzu", "variations": ["japanese", "korean"], "intensity_range": (6, 8)},
                {"base_name": "citron", "variations": ["corsican", "moroccan"], "intensity_range": (7, 9)}
            ],
            
            # 꽃류
            "floral_absolutes": [
                {"base_name": "rose", "variations": ["damascena", "centifolia", "tea"], "intensity_range": (7, 10)},
                {"base_name": "jasmine", "variations": ["grandiflorum", "sambac", "auriculatum"], "intensity_range": (9, 10)},
                {"base_name": "lavender", "variations": ["french", "bulgarian", "english"], "intensity_range": (5, 7)},
                {"base_name": "ylang_ylang", "variations": ["extra", "first", "complete"], "intensity_range": (8, 10)},
                {"base_name": "tuberose", "variations": ["indian", "french"], "intensity_range": (9, 10)},
                {"base_name": "neroli", "variations": ["tunisia", "morocco"], "intensity_range": (6, 8)},
                {"base_name": "geranium", "variations": ["bourbon", "egypt", "china"], "intensity_range": (6, 8)},
                {"base_name": "violet", "variations": ["leaf", "flower"], "intensity_range": (4, 6)}
            ],
            
            # 목재류
            "wood_extracts": [
                {"base_name": "sandalwood", "variations": ["mysore", "australian", "hawaiian"], "intensity_range": (6, 9)},
                {"base_name": "cedar", "variations": ["atlas", "virginian", "texas"], "intensity_range": (7, 9)},
                {"base_name": "rosewood", "variations": ["brazilian", "indian"], "intensity_range": (5, 7)},
                {"base_name": "agarwood", "variations": ["cambodian", "indian", "synthetic"], "intensity_range": (9, 10)},
                {"base_name": "ebony", "variations": ["african", "indian"], "intensity_range": (8, 10)},
                {"base_name": "birch", "variations": ["tar", "sweet"], "intensity_range": (7, 9)}
            ],
            
            # 향신료
            "spice_oils": [
                {"base_name": "cinnamon", "variations": ["bark", "leaf", "cassia"], "intensity_range": (8, 10)},
                {"base_name": "clove", "variations": ["bud", "leaf"], "intensity_range": (9, 10)},
                {"base_name": "cardamom", "variations": ["green", "black"], "intensity_range": (6, 8)},
                {"base_name": "ginger", "variations": ["fresh", "dried"], "intensity_range": (7, 9)},
                {"base_name": "nutmeg", "variations": ["east_indian", "west_indian"], "intensity_range": (7, 9)},
                {"base_name": "black_pepper", "variations": ["madagascar", "indian"], "intensity_range": (8, 10)}
            ],
            
            # 허브류
            "herbal_extracts": [
                {"base_name": "basil", "variations": ["sweet", "holy", "lemon"], "intensity_range": (6, 8)},
                {"base_name": "rosemary", "variations": ["spanish", "french", "tunisian"], "intensity_range": (7, 9)},
                {"base_name": "thyme", "variations": ["red", "white"], "intensity_range": (8, 10)},
                {"base_name": "mint", "variations": ["peppermint", "spearmint", "apple"], "intensity_range": (8, 10)},
                {"base_name": "sage", "variations": ["clary", "common"], "intensity_range": (6, 8)}
            ],
            
            # 수지류
            "resin_materials": [
                {"base_name": "frankincense", "variations": ["boswellia_sacra", "carterii"], "intensity_range": (7, 9)},
                {"base_name": "myrrh", "variations": ["commiphora_myrrha", "sweet"], "intensity_range": (8, 10)},
                {"base_name": "benzoin", "variations": ["siam", "sumatra"], "intensity_range": (6, 8)},
                {"base_name": "labdanum", "variations": ["cistus", "rockrose"], "intensity_range": (8, 10)},
                {"base_name": "elemi", "variations": ["filipino", "mexican"], "intensity_range": (5, 7)}
            ]
        }
    
    def _initialize_synthetic_materials(self) -> Dict:
        """합성 원료 기본 템플릿"""
        return {
            # 알데하이드류
            "aldehydes": [
                {"base_name": "aldehyde_c10", "cas_number": "112-31-2", "intensity_range": (8, 10)},
                {"base_name": "aldehyde_c11", "cas_number": "112-44-7", "intensity_range": (7, 9)},
                {"base_name": "aldehyde_c12", "cas_number": "112-54-9", "intensity_range": (6, 8)},
                {"base_name": "benzaldehyde", "cas_number": "100-52-7", "intensity_range": (7, 9)},
                {"base_name": "vanillin", "cas_number": "121-33-5", "intensity_range": (8, 10)},
                {"base_name": "ethyl_vanillin", "cas_number": "121-32-4", "intensity_range": (9, 10)}
            ],
            
            # 에스테르류
            "esters": [
                {"base_name": "linalyl_acetate", "cas_number": "115-95-7", "intensity_range": (5, 7)},
                {"base_name": "benzyl_acetate", "cas_number": "140-11-4", "intensity_range": (6, 8)},
                {"base_name": "geranyl_acetate", "cas_number": "105-87-3", "intensity_range": (6, 8)},
                {"base_name": "phenylethyl_acetate", "cas_number": "103-45-7", "intensity_range": (7, 9)},
                {"base_name": "methyl_anthranilate", "cas_number": "134-20-3", "intensity_range": (8, 10)}
            ],
            
            # 알코올류
            "alcohols": [
                {"base_name": "linalool", "cas_number": "78-70-6", "intensity_range": (5, 7)},
                {"base_name": "geraniol", "cas_number": "106-24-1", "intensity_range": (6, 8)},
                {"base_name": "citronellol", "cas_number": "106-22-9", "intensity_range": (5, 7)},
                {"base_name": "benzyl_alcohol", "cas_number": "100-51-6", "intensity_range": (4, 6)},
                {"base_name": "phenylethyl_alcohol", "cas_number": "60-12-8", "intensity_range": (6, 8)}
            ],
            
            # 케톤류
            "ketones": [
                {"base_name": "iso_e_super", "cas_number": "54464-57-2", "intensity_range": (3, 5)},
                {"base_name": "cashmeran", "cas_number": "33704-61-9", "intensity_range": (4, 6)},
                {"base_name": "damascone", "cas_number": "23726-93-4", "intensity_range": (7, 9)},
                {"base_name": "ionone", "cas_number": "8013-90-9", "intensity_range": (6, 8)},
                {"base_name": "muscone", "cas_number": "541-91-3", "intensity_range": (8, 10)}
            ],
            
            # 락톤류
            "lactones": [
                {"base_name": "gamma_decalactone", "cas_number": "706-14-9", "intensity_range": (7, 9)},
                {"base_name": "gamma_undecalactone", "cas_number": "104-67-6", "intensity_range": (8, 10)},
                {"base_name": "delta_dodecalactone", "cas_number": "713-95-1", "intensity_range": (6, 8)},
                {"base_name": "coumarin", "cas_number": "91-64-5", "intensity_range": (7, 9)}
            ],
            
            # 아로마틱 화합물
            "aromatics": [
                {"base_name": "amyl_salicylate", "cas_number": "2050-08-0", "intensity_range": (5, 7)},
                {"base_name": "iso_bornyl_acetate", "cas_number": "125-12-2", "intensity_range": (4, 6)},
                {"base_name": "dihydromyrcenol", "cas_number": "18479-58-8", "intensity_range": (6, 8)},
                {"base_name": "hedione", "cas_number": "24851-98-7", "intensity_range": (4, 6)}
            ]
        }
    
    def generate_material_entry(self, material_type: str, base_data: Dict, family: str, 
                              is_natural: bool, index: int) -> Dict:
        """단일 향료 원료 데이터 생성"""
        
        # 기본 정보 설정
        if is_natural:
            variations = base_data.get("variations", ["standard"])
            variation = random.choice(variations)
            material_name = f"{base_data['base_name']}_{variation}_{family}"
            extraction_method = random.choice([
                "steam_distillation", "cold_press", "solvent_extraction", 
                "co2_extraction", "enfluerage"
            ])
        else:
            material_name = f"{base_data['base_name']}_synthetic_{family}"
            extraction_method = random.choice(["chemical_synthesis", "biotechnology"])
        
        # 물리화학적 특성
        intensity_range = base_data["intensity_range"]
        base_intensity = random.uniform(*intensity_range)
        
        # 휘발성 결정
        if base_intensity >= 8:
            volatility = "top"
            longevity = random.uniform(0.5, 3.0)
        elif base_intensity >= 6:
            volatility = "middle" 
            longevity = random.uniform(2.0, 8.0)
        else:
            volatility = "base"
            longevity = random.uniform(6.0, 24.0)
        
        # 가격 및 가용성 (자연/합성에 따라)
        if is_natural:
            base_price = random.uniform(50, 500)  # $/kg
            availability = random.uniform(0.3, 0.9)
        else:
            base_price = random.uniform(10, 150)  # $/kg
            availability = random.uniform(0.8, 1.0)
        
        # 추출 수율
        if is_natural:
            yield_percentage = random.uniform(0.1, 15.0)
        else:
            yield_percentage = random.uniform(85.0, 99.5)
        
        # 화학적 특성
        boiling_point = random.uniform(80, 350)
        density = random.uniform(0.8, 1.2)
        refractive_index = random.uniform(1.400, 1.600)
        
        # 관능적 특성
        odor_descriptors = self._generate_odor_descriptors(family, is_natural)
        
        # 지역별 변이 (자연 원료만)
        origin_country = ""
        if is_natural:
            regions = ["France", "Bulgaria", "India", "Morocco", "Italy", "Greece", 
                      "Turkey", "Egypt", "Madagascar", "Indonesia", "China", "Brazil"]
            origin_country = random.choice(regions)
        
        # 계절성 및 수확 정보 (자연 원료만)
        harvest_info = {}
        if is_natural:
            harvest_info = {
                "best_harvest_month": random.randint(1, 12),
                "shelf_life_months": random.randint(12, 48),
                "storage_temp_c": random.randint(-5, 25)
            }
        
        return {
            "id": f"FRAG_{index:06d}",
            "name": material_name,
            "family": family,
            "is_natural": is_natural,
            "origin_type": "natural" if is_natural else "synthetic",
            
            # 화학 정보
            "cas_number": base_data.get("cas_number", ""),
            "molecular_formula": self._generate_molecular_formula(),
            "molecular_weight": random.uniform(100, 300),
            
            # 물리적 특성
            "physical_properties": {
                "boiling_point_c": round(boiling_point, 1),
                "density_g_ml": round(density, 3),
                "refractive_index": round(refractive_index, 4),
                "flash_point_c": random.randint(50, 150),
                "solubility_ethanol": random.uniform(0.1, 100.0)
            },
            
            # 향기 특성  
            "olfactory_properties": {
                "intensity": round(base_intensity, 1),
                "longevity_hours": round(longevity, 1),
                "volatility": volatility,
                "diffusion": random.uniform(1, 10),
                "odor_descriptors": odor_descriptors,
                "threshold_ppb": random.uniform(0.001, 100.0)
            },
            
            # 추출/제조 정보
            "production_info": {
                "extraction_method": extraction_method,
                "yield_percentage": round(yield_percentage, 2),
                "purity_percentage": round(self.extraction_methods[extraction_method]["purity"] * 100, 1),
                "processing_time_hours": random.randint(2, 72)
            },
            
            # 경제 정보
            "economic_data": {
                "price_usd_per_kg": round(base_price, 2),
                "availability_score": round(availability, 2),
                "annual_production_tons": random.randint(10, 10000),
                "main_suppliers": random.sample(["BASF", "Firmenich", "IFF", "Givaudan", 
                                               "Symrise", "T.Hasegawa", "Mane"], k=2)
            },
            
            # 지역 정보 (자연 원료만)
            "geographical_info": {
                "origin_country": origin_country,
                "cultivation_regions": [origin_country] if origin_country else [],
                "climate_requirements": random.choice(["tropical", "temperate", "arid", "mediterranean"]) if is_natural else ""
            },
            
            # 수확/저장 정보
            "harvest_storage": harvest_info,
            
            # 규제 정보
            "regulatory_info": {
                "ifra_restricted": random.choice([True, False]),
                "max_concentration_percent": random.uniform(0.1, 10.0),
                "allergen_declaration": random.choice([True, False]),
                "natural_complex": is_natural
            },
            
            # 응용 분야
            "applications": {
                "fine_fragrance": random.choice([True, False]),
                "personal_care": random.choice([True, False]),
                "home_care": random.choice([True, False]),
                "air_care": random.choice([True, False])
            },
            
            # 메타데이터
            "metadata": {
                "created_date": "2025-01-15",
                "data_version": "2.0",
                "quality_grade": random.choice(["A", "B", "C"]),
                "research_notes": f"Generated material for {family} family training"
            }
        }
    
    def _generate_odor_descriptors(self, family: str, is_natural: bool) -> List[str]:
        """향기 계열별 향취 설명자 생성"""
        descriptors_by_family = {
            "citrus": ["fresh", "zesty", "bright", "sparkling", "tangy", "juicy"],
            "floral": ["rosy", "powdery", "sweet", "romantic", "elegant", "delicate"],
            "woody": ["warm", "dry", "creamy", "smoky", "balsamic", "rich"],
            "oriental": ["spicy", "exotic", "mysterious", "luxurious", "sensual", "warm"],
            "fresh": ["clean", "aquatic", "ozonic", "crisp", "transparent", "airy"],
            "fruity": ["juicy", "sweet", "ripe", "tropical", "nectar-like", "succulent"],
            "green": ["leafy", "crushed", "herbal", "dewy", "natural", "verdant"],
            "aquatic": ["marine", "salty", "oceanic", "misty", "wet", "coastal"],
            "spicy": ["warm", "peppery", "aromatic", "pungent", "exotic", "stimulating"],
            "gourmand": ["edible", "creamy", "sweet", "comforting", "delicious", "indulgent"],
            "herbal": ["medicinal", "camphoraceous", "cooling", "therapeutic", "green", "dry"],
            "mineral": ["metallic", "stony", "earthy", "dusty", "chalky", "concrete"],
            "smoky": ["burnt", "tarry", "leathery", "tobacco", "fire", "charred"]
        }
        
        base_descriptors = descriptors_by_family.get(family, ["pleasant", "distinctive"])
        
        # 자연 원료는 더 복잡한 설명자
        if is_natural:
            additional = ["complex", "nuanced", "multifaceted"]
        else:
            additional = ["clean", "precise", "linear"]
        
        selected = random.sample(base_descriptors, k=min(3, len(base_descriptors)))
        selected.extend(random.sample(additional, k=1))
        
        return selected[:4]
    
    def _generate_molecular_formula(self) -> str:
        """분자식 생성 (단순화)"""
        c_count = random.randint(5, 20)
        h_count = random.randint(8, 40)
        
        # 추가 원소 (30% 확률)
        if random.random() < 0.3:
            extra_elements = random.choice(["O", "O2", "N", "S"])
            return f"C{c_count}H{h_count}{extra_elements}"
        else:
            return f"C{c_count}H{h_count}"
    
    def generate_dataset(self, total_count: int = 200000) -> None:
        """전체 데이터셋 생성"""
        print(f"Starting generation of {total_count:,} fragrance materials dataset...")
        
        dataset = []
        generated_count = 0
        
        for family, family_info in self.scent_families.items():
            # 계열별 생성할 개수 계산
            family_count = int(total_count * family_info["weight"])
            natural_count = int(family_count * family_info["natural_ratio"])
            synthetic_count = family_count - natural_count
            
            print(f"Family {family.title()}: {family_count:,} materials (Natural {natural_count:,}, Synthetic {synthetic_count:,})")
            
            # 자연 원료 생성
            for _ in range(natural_count):
                # 랜덤하게 자연 원료 카테고리 선택
                category = random.choice(list(self.natural_materials.keys()))
                material_list = self.natural_materials[category]
                base_data = random.choice(material_list)
                
                entry = self.generate_material_entry(
                    material_type="natural",
                    base_data=base_data,
                    family=family,
                    is_natural=True,
                    index=generated_count
                )
                
                dataset.append(entry)
                generated_count += 1
                
                if generated_count % 10000 == 0:
                    print(f"Progress: {generated_count:,}/{total_count:,} ({(generated_count/total_count)*100:.1f}%)")
            
            # 합성 원료 생성
            for _ in range(synthetic_count):
                # 랜덤하게 합성 원료 카테고리 선택
                category = random.choice(list(self.synthetic_materials.keys()))
                material_list = self.synthetic_materials[category]
                base_data = random.choice(material_list)
                
                entry = self.generate_material_entry(
                    material_type="synthetic",
                    base_data=base_data,
                    family=family,
                    is_natural=False,
                    index=generated_count
                )
                
                dataset.append(entry)
                generated_count += 1
                
                if generated_count % 10000 == 0:
                    print(f"Progress: {generated_count:,}/{total_count:,} ({(generated_count/total_count)*100:.1f}%)")
        
        # 최종 조정 (정확히 200,000개 맞추기)
        while len(dataset) < total_count:
            family = random.choice(list(self.scent_families.keys()))
            is_natural = random.random() < 0.5
            
            if is_natural:
                category = random.choice(list(self.natural_materials.keys()))
                material_list = self.natural_materials[category]
            else:
                category = random.choice(list(self.synthetic_materials.keys()))
                material_list = self.synthetic_materials[category]
            
            base_data = random.choice(material_list)
            entry = self.generate_material_entry(
                material_type="natural" if is_natural else "synthetic",
                base_data=base_data,
                family=family,
                is_natural=is_natural,
                index=generated_count
            )
            
            dataset.append(entry)
            generated_count += 1
        
        # 데이터셋 무작위 섞기
        random.shuffle(dataset)
        
        print(f"Total {len(dataset):,} materials generated!")
        
        # 통계 출력
        self._print_dataset_statistics(dataset)
        
        # 파일로 저장
        self._save_dataset(dataset)
    
    def _print_dataset_statistics(self, dataset: List[Dict]) -> None:
        """데이터셋 통계 출력"""
        print("\nDataset Statistics:")
        
        # 자연/합성 비율
        natural_count = sum(1 for item in dataset if item["is_natural"])
        synthetic_count = len(dataset) - natural_count
        print(f"- Natural materials: {natural_count:,} ({(natural_count/len(dataset)*100):.1f}%)")
        print(f"- Synthetic materials: {synthetic_count:,} ({(synthetic_count/len(dataset)*100):.1f}%)")
        
        # 계열별 분포
        family_counts = {}
        for item in dataset:
            family = item["family"]
            family_counts[family] = family_counts.get(family, 0) + 1
        
        print("\nScent Family Distribution:")
        for family, count in sorted(family_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"- {family.title()}: {count:,} ({(count/len(dataset)*100):.1f}%)")
        
        # 휘발성 분포
        volatility_counts = {"top": 0, "middle": 0, "base": 0}
        for item in dataset:
            vol = item["olfactory_properties"]["volatility"]
            volatility_counts[vol] += 1
        
        print("\nVolatility Distribution:")
        for vol, count in volatility_counts.items():
            print(f"- {vol.title()} note: {count:,} ({(count/len(dataset)*100):.1f}%)")
        
        # 가격 분포
        prices = [item["economic_data"]["price_usd_per_kg"] for item in dataset]
        print(f"\nPrice Distribution:")
        print(f"- Average price: ${np.mean(prices):.2f}/kg")
        print(f"- Median price: ${np.median(prices):.2f}/kg") 
        print(f"- Price range: ${min(prices):.2f} - ${max(prices):.2f}/kg")
    
    def _save_dataset(self, dataset: List[Dict]) -> None:
        """데이터셋을 여러 형태로 저장"""
        print("\nSaving dataset...")
        
        # 1. 전체 JSON 저장
        json_file = self.output_dir / "fragrance_materials_200k.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_count": len(dataset),
                    "created_date": "2025-01-15",
                    "version": "2.0",
                    "description": "Comprehensive fragrance raw materials dataset for deep learning"
                },
                "materials": dataset
            }, f, indent=2, ensure_ascii=False)
        
        print(f"JSON file saved: {json_file}")
        
        # 2. CSV 형태 저장 (간소화된 형태)
        csv_file = self.output_dir / "fragrance_materials_200k.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'id', 'name', 'family', 'is_natural', 'origin_type',
                'intensity', 'longevity_hours', 'volatility',
                'price_usd_per_kg', 'availability_score',
                'extraction_method', 'yield_percentage',
                'boiling_point_c', 'molecular_weight'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in dataset:
                row = {
                    'id': item['id'],
                    'name': item['name'],
                    'family': item['family'],
                    'is_natural': item['is_natural'],
                    'origin_type': item['origin_type'],
                    'intensity': item['olfactory_properties']['intensity'],
                    'longevity_hours': item['olfactory_properties']['longevity_hours'],
                    'volatility': item['olfactory_properties']['volatility'],
                    'price_usd_per_kg': item['economic_data']['price_usd_per_kg'],
                    'availability_score': item['economic_data']['availability_score'],
                    'extraction_method': item['production_info']['extraction_method'],
                    'yield_percentage': item['production_info']['yield_percentage'],
                    'boiling_point_c': item['physical_properties']['boiling_point_c'],
                    'molecular_weight': item['molecular_weight']
                }
                writer.writerow(row)
        
        print(f"CSV file saved: {csv_file}")
        
        # 3. 딥러닝 훈련용 분할 저장
        train_size = int(len(dataset) * 0.8)
        val_size = int(len(dataset) * 0.1)
        
        train_data = dataset[:train_size]
        val_data = dataset[train_size:train_size + val_size]
        test_data = dataset[train_size + val_size:]
        
        # 훈련 세트
        train_file = self.output_dir / "fragrance_train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump({"materials": train_data}, f, indent=2, ensure_ascii=False)
        
        # 검증 세트
        val_file = self.output_dir / "fragrance_validation.json"
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump({"materials": val_data}, f, indent=2, ensure_ascii=False)
        
        # 테스트 세트
        test_file = self.output_dir / "fragrance_test.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump({"materials": test_data}, f, indent=2, ensure_ascii=False)
        
        print(f"Training sets saved:")
        print(f"   - Training: {len(train_data):,} ({len(train_data)/len(dataset)*100:.1f}%)")
        print(f"   - Validation: {len(val_data):,} ({len(val_data)/len(dataset)*100:.1f}%)")
        print(f"   - Test: {len(test_data):,} ({len(test_data)/len(dataset)*100:.1f}%)")
        
        print(f"\nDataset generation completed!")
        print(f"Saved to: {self.output_dir.absolute()}")

if __name__ == "__main__":
    generator = FragranceDatasetGenerator()
    generator.generate_dataset(200000)