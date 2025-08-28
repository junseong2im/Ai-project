from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from PIL import Image

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Chemical structure processing disabled.")

try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Image processing will be limited.")


@dataclass
class FragranceProperties:
    """향료의 다차원적 특성"""
    # 화학적 특성
    molecular_weight: float
    boiling_point: Optional[float]
    volatility_rate: int  # 1-10 scale
    solubility_water: float  # 0-1 scale
    solubility_alcohol: float  # 0-1 scale
    
    # 후각적 특성
    odor_intensity: float  # 0-10 scale  
    odor_family: List[str]  # ['citrus', 'floral', 'woody', etc.]
    odor_description: List[str]  # ['fresh', 'sweet', 'sharp', etc.]
    
    # 감성적 특성
    emotional_associations: List[str]  # ['calm', 'energetic', 'romantic', etc.]
    cultural_associations: List[str]  # ['western', 'eastern', 'modern', etc.]
    
    # 사용 특성
    preferred_season: List[str]  # ['spring', 'summer', 'autumn', 'winter']
    preferred_time: List[str]  # ['morning', 'afternoon', 'evening', 'night']
    longevity: int  # 1-10 scale
    projection: int  # 1-10 scale
    
    # 화학 구조 (선택적)
    smiles: Optional[str] = None
    molecular_fingerprint: Optional[List[int]] = None


@dataclass
class MultimodalInput:
    """다중 모달 입력 데이터"""
    text_description: str
    image_path: Optional[str] = None
    fragrance_properties: Optional[Dict[str, FragranceProperties]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    contextual_info: Optional[Dict[str, Any]] = None


class ChemicalFeatureExtractor:
    """화학 구조에서 특성 추출"""
    
    def __init__(self):
        self.available = RDKIT_AVAILABLE
        
    def smiles_to_features(self, smiles: str) -> Dict[str, float]:
        """SMILES 표기법으로부터 분자 특성 추출"""
        if not self.available:
            return self._get_default_chemical_features()
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._get_default_chemical_features()
                
            features = {
                # 기본 분자 특성
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),  # 지용성
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_rings': Descriptors.RingCount(mol),
                
                # 향료 관련 특성
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'num_heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'tpsa': Descriptors.TPSA(mol),  # 극성 표면적
                
                # 후각 관련 예측 특성
                'volatility_score': self._predict_volatility(mol),
                'intensity_score': self._predict_intensity(mol),
                'longevity_score': self._predict_longevity(mol)
            }
            
            return features
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return self._get_default_chemical_features()
    
    def _predict_volatility(self, mol) -> float:
        """분자 구조로부터 휘발성 예측"""
        # 분자량과 boiling point 추정 기반 휴리스틱
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        # 낮은 분자량과 적절한 lipophilicity는 높은 휘발성을 의미
        volatility = max(0, min(10, (300 - mw) / 30 + (3 - abs(logp - 2)) * 2))
        return volatility
    
    def _predict_intensity(self, mol) -> float:
        """분자 구조로부터 향의 강도 예측"""
        # 방향족 고리와 특정 관능기 기반
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        hetero_atoms = Descriptors.NumHeteroatoms(mol)
        
        # 방향족 고리가 많을수록, 헤테로 원자가 많을수록 강한 향
        intensity = min(10, aromatic_rings * 2 + hetero_atoms * 0.5)
        return intensity
    
    def _predict_longevity(self, mol) -> float:
        """분자 구조로부터 지속성 예측"""
        # 분자량과 극성 표면적 기반
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # 높은 분자량과 적절한 극성은 긴 지속성을 의미
        longevity = min(10, (mw - 100) / 50 + (100 - abs(tpsa - 60)) / 20)
        return max(0, longevity)
    
    def get_molecular_fingerprint(self, smiles: str) -> List[int]:
        """분자 지문 생성"""
        if not self.available:
            return [0] * 1024  # 기본값
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [0] * 1024
                
            # Morgan fingerprint (ECFP) 생성
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            return [int(x) for x in fp.ToBitString()]
            
        except Exception:
            return [0] * 1024
    
    def _get_default_chemical_features(self) -> Dict[str, float]:
        """RDKit 없을 때 기본 화학 특성"""
        return {
            'molecular_weight': 150.0,
            'logp': 2.0,
            'num_rotatable_bonds': 3,
            'num_rings': 1,
            'num_aromatic_rings': 1,
            'num_heavy_atoms': 12,
            'tpsa': 40.0,
            'volatility_score': 5.0,
            'intensity_score': 5.0,
            'longevity_score': 5.0
        }


class ImageFeatureExtractor:
    """이미지에서 특성 추출 (향수병, 원료 이미지 등)"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.available = CLIP_AVAILABLE
        
        if self.available:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
    def extract_image_features(self, image_path: str) -> torch.Tensor:
        """이미지에서 특성 벡터 추출"""
        if not self.available:
            return torch.zeros(512)  # CLIP 기본 차원
            
        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            return image_features.cpu().squeeze()
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return torch.zeros(512)
    
    def get_image_text_similarity(self, image_path: str, text_description: str) -> float:
        """이미지와 텍스트 간 유사도 계산"""
        if not self.available:
            return 0.5  # 기본값
            
        try:
            image = Image.open(image_path)
            inputs = self.processor(
                text=[text_description], 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                similarity = torch.softmax(logits_per_image, dim=-1).cpu().numpy()[0, 0]
                
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating image-text similarity: {e}")
            return 0.5


class MultiModalFragranceEncoder(nn.Module):
    """다중 모달 향료 인코더"""
    
    def __init__(
        self,
        text_dim: int = 768,  # BERT 임베딩 차원
        chemical_feature_dim: int = 10,
        image_dim: int = 512,  # CLIP 이미지 특성 차원
        output_dim: int = 512,
        num_attention_heads: int = 8
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.chemical_feature_dim = chemical_feature_dim
        self.image_dim = image_dim
        self.output_dim = output_dim
        
        # 각 모달리티를 공통 차원으로 투영
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.chemical_projection = nn.Sequential(
            nn.Linear(chemical_feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 분자 지문용 별도 인코더
        self.fingerprint_encoder = nn.Sequential(
            nn.Linear(1024, 256),  # 분자 지문은 1024 비트
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Cross-modal attention
        self.cross_attention = MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 모달리티 가중치 학습
        self.modality_weights = nn.Parameter(torch.ones(4))  # text, chemical, image, fingerprint
        
        # 최종 융합 레이어
        self.fusion_layers = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(
        self,
        text_features: Optional[torch.Tensor] = None,
        chemical_features: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        molecular_fingerprint: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        다중 모달 특성 융합
        
        Args:
            text_features: (batch_size, text_dim)
            chemical_features: (batch_size, chemical_feature_dim)  
            image_features: (batch_size, image_dim)
            molecular_fingerprint: (batch_size, 1024)
            mask: 사용 가능한 모달리티 마스크
        
        Returns:
            fused_features: (batch_size, output_dim)
            attention_weights: 각 모달리티의 가중치
        """
        batch_size = None
        projected_features = []
        modality_names = []
        
        # 텍스트 특성 처리
        if text_features is not None:
            batch_size = text_features.shape[0]
            text_proj = self.text_projection(text_features)
            projected_features.append(text_proj)
            modality_names.append('text')
        
        # 화학적 특성 처리
        if chemical_features is not None:
            if batch_size is None:
                batch_size = chemical_features.shape[0]
            chem_proj = self.chemical_projection(chemical_features)
            projected_features.append(chem_proj)
            modality_names.append('chemical')
        
        # 이미지 특성 처리
        if image_features is not None:
            if batch_size is None:
                batch_size = image_features.shape[0]
            img_proj = self.image_projection(image_features)
            projected_features.append(img_proj)
            modality_names.append('image')
        
        # 분자 지문 처리
        if molecular_fingerprint is not None:
            if batch_size is None:
                batch_size = molecular_fingerprint.shape[0]
            fp_proj = self.fingerprint_encoder(molecular_fingerprint.float())
            projected_features.append(fp_proj)
            modality_names.append('fingerprint')
        
        if not projected_features:
            # 모든 모달리티가 None인 경우 기본값 반환
            return torch.zeros(1, self.output_dim), {}
        
        # 모든 특성을 시퀀스로 결합
        stacked_features = torch.stack(projected_features, dim=1)  # (batch, num_modalities, output_dim)
        
        # Self-attention으로 모달리티 간 상호작용
        attended_features, attention_weights = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # 모달리티별 가중치 적용
        weights = F.softmax(self.modality_weights[:len(projected_features)], dim=0)
        weighted_features = attended_features * weights.view(1, -1, 1)
        
        # 가중 평균으로 융합
        fused_features = weighted_features.mean(dim=1)  # (batch, output_dim)
        
        # 최종 변환
        final_features = self.fusion_layers(fused_features)
        
        # 어텐션 가중치 정보 반환
        attention_info = {
            modality: float(weights[i])
            for i, modality in enumerate(modality_names)
        }
        
        return final_features, attention_info


class FragranceMultiModalSystem:
    """향료 다중 모달 통합 시스템"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        fragrance_db_path: str = "data/fragrance_properties.json"
    ):
        # 특성 추출기들 초기화
        self.chemical_extractor = ChemicalFeatureExtractor()
        self.image_extractor = ImageFeatureExtractor()
        
        # 다중 모달 인코더 초기화
        self.encoder = MultiModalFragranceEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        
        # 향료 속성 데이터베이스
        self.fragrance_db_path = Path(fragrance_db_path)
        self.fragrance_properties = self._load_fragrance_properties()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def _load_fragrance_properties(self) -> Dict[str, FragranceProperties]:
        """향료 속성 데이터베이스 로드"""
        if self.fragrance_db_path.exists():
            try:
                with open(self.fragrance_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                properties = {}
                for name, props in data.items():
                    properties[name] = FragranceProperties(**props)
                
                return properties
            except Exception as e:
                print(f"Error loading fragrance properties: {e}")
        
        # 기본 샘플 데이터 생성
        return self._create_sample_fragrance_properties()
    
    def _create_sample_fragrance_properties(self) -> Dict[str, FragranceProperties]:
        """샘플 향료 속성 데이터 생성"""
        sample_properties = {
            "베르가못": FragranceProperties(
                molecular_weight=176.25,
                boiling_point=211.0,
                volatility_rate=9,
                solubility_water=0.1,
                solubility_alcohol=0.9,
                odor_intensity=8.0,
                odor_family=["citrus", "fresh"],
                odor_description=["bright", "zesty", "uplifting"],
                emotional_associations=["energetic", "refreshing", "optimistic"],
                cultural_associations=["mediterranean", "tea", "morning"],
                preferred_season=["spring", "summer"],
                preferred_time=["morning", "afternoon"],
                longevity=3,
                projection=7,
                smiles="CC1=CC2=C(C(C)(C)CCC2=C(C)C1=O)CO"
            ),
            "라벤더": FragranceProperties(
                molecular_weight=154.25,
                boiling_point=188.0,
                volatility_rate=6,
                solubility_water=0.2,
                solubility_alcohol=0.8,
                odor_intensity=6.0,
                odor_family=["floral", "herbal"],
                odor_description=["soothing", "clean", "powdery"],
                emotional_associations=["calming", "peaceful", "nostalgic"],
                cultural_associations=["french", "countryside", "relaxation"],
                preferred_season=["spring", "autumn"],
                preferred_time=["evening", "night"],
                longevity=5,
                projection=4,
                smiles="CC1=CC=C(C(C)(C)O)C=C1"
            ),
            "바닐라": FragranceProperties(
                molecular_weight=152.15,
                boiling_point=285.0,
                volatility_rate=2,
                solubility_water=0.05,
                solubility_alcohol=0.7,
                odor_intensity=9.0,
                odor_family=["gourmand", "sweet"],
                odor_description=["warm", "creamy", "comforting"],
                emotional_associations=["comforting", "sensual", "nostalgic"],
                cultural_associations=["tropical", "dessert", "warmth"],
                preferred_season=["autumn", "winter"],
                preferred_time=["evening", "night"],
                longevity=8,
                projection=6,
                smiles="COC1=CC(C=O)=CC=C1O"
            )
        }
        
        # 분자 지문 생성
        for name, props in sample_properties.items():
            if props.smiles:
                props.molecular_fingerprint = self.chemical_extractor.get_molecular_fingerprint(props.smiles)
        
        return sample_properties
    
    def encode_multimodal_input(self, input_data: MultimodalInput) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """다중 모달 입력을 인코딩"""
        # 텍스트 특성 (여기서는 기본 임베딩 사용, 실제로는 BERT 등 사용)
        text_features = None
        if input_data.text_description:
            # 간단한 텍스트 특성 생성 (실제로는 사전 훈련된 모델 사용)
            text_features = torch.randn(1, 768)  # BERT 차원
        
        # 이미지 특성
        image_features = None
        if input_data.image_path:
            img_feat = self.image_extractor.extract_image_features(input_data.image_path)
            if img_feat.numel() > 0:
                image_features = img_feat.unsqueeze(0)
        
        # 화학적 특성 및 분자 지문
        chemical_features = None
        molecular_fingerprint = None
        
        if input_data.fragrance_properties:
            # 여러 향료의 특성을 평균내어 하나의 벡터로 생성
            all_chem_features = []
            all_fingerprints = []
            
            for name, props in input_data.fragrance_properties.items():
                if props.smiles:
                    chem_feat = self.chemical_extractor.smiles_to_features(props.smiles)
                    chem_vector = torch.tensor([
                        chem_feat['molecular_weight'] / 300.0,  # 정규화
                        chem_feat['logp'] / 5.0,
                        chem_feat['volatility_score'] / 10.0,
                        chem_feat['intensity_score'] / 10.0,
                        chem_feat['longevity_score'] / 10.0,
                        props.volatility_rate / 10.0,
                        props.odor_intensity / 10.0,
                        props.longevity / 10.0,
                        props.projection / 10.0,
                        props.solubility_alcohol
                    ])
                    all_chem_features.append(chem_vector)
                    
                    if props.molecular_fingerprint:
                        fp_tensor = torch.tensor(props.molecular_fingerprint, dtype=torch.float)
                        all_fingerprints.append(fp_tensor)
            
            if all_chem_features:
                chemical_features = torch.stack(all_chem_features).mean(dim=0, keepdim=True)
            
            if all_fingerprints:
                molecular_fingerprint = torch.stack(all_fingerprints).mean(dim=0, keepdim=True)
        
        # 입력을 디바이스로 이동
        if text_features is not None:
            text_features = text_features.to(self.device)
        if chemical_features is not None:
            chemical_features = chemical_features.to(self.device)
        if image_features is not None:
            image_features = image_features.to(self.device)
        if molecular_fingerprint is not None:
            molecular_fingerprint = molecular_fingerprint.to(self.device)
        
        # 다중 모달 인코딩 수행
        encoded_features, attention_weights = self.encoder(
            text_features=text_features,
            chemical_features=chemical_features,
            image_features=image_features,
            molecular_fingerprint=molecular_fingerprint
        )
        
        # 메타데이터 생성
        metadata = {
            'attention_weights': attention_weights,
            'modalities_used': [],
            'encoding_confidence': self._calculate_encoding_confidence(attention_weights)
        }
        
        if text_features is not None:
            metadata['modalities_used'].append('text')
        if chemical_features is not None:
            metadata['modalities_used'].append('chemical')
        if image_features is not None:
            metadata['modalities_used'].append('image')
        if molecular_fingerprint is not None:
            metadata['modalities_used'].append('fingerprint')
        
        return encoded_features, metadata
    
    def _calculate_encoding_confidence(self, attention_weights: Dict[str, float]) -> float:
        """인코딩 신뢰도 계산"""
        if not attention_weights:
            return 0.0
        
        # 사용된 모달리티 수와 가중치 분산 기반 신뢰도
        num_modalities = len(attention_weights)
        weight_variance = np.var(list(attention_weights.values()))
        
        # 많은 모달리티 사용 + 균등한 가중치 = 높은 신뢰도
        modality_bonus = min(num_modalities / 4.0, 1.0)  # 최대 4개 모달리티
        balance_bonus = max(0, 1.0 - weight_variance * 2)  # 가중치가 균등할수록 좋음
        
        confidence = (modality_bonus + balance_bonus) / 2
        return min(confidence, 1.0)
    
    def get_fragrance_similarity(
        self, 
        input1: MultimodalInput, 
        input2: MultimodalInput
    ) -> Tuple[float, Dict[str, Any]]:
        """두 향료 입력 간의 유사도 계산"""
        
        # 각각 인코딩
        features1, meta1 = self.encode_multimodal_input(input1)
        features2, meta2 = self.encode_multimodal_input(input2)
        
        # 코사인 유사도 계산
        similarity = F.cosine_similarity(features1, features2, dim=1).item()
        
        # 상세 정보
        details = {
            'similarity_score': similarity,
            'input1_modalities': meta1['modalities_used'],
            'input2_modalities': meta2['modalities_used'],
            'input1_confidence': meta1['encoding_confidence'],
            'input2_confidence': meta2['encoding_confidence'],
            'overall_confidence': (meta1['encoding_confidence'] + meta2['encoding_confidence']) / 2
        }
        
        return similarity, details
    
    def save_model(self, path: str) -> None:
        """모델 저장"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'fragrance_properties': self.fragrance_properties
        }, path)
    
    def load_model(self, path: str) -> None:
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if 'fragrance_properties' in checkpoint:
            self.fragrance_properties = checkpoint['fragrance_properties']
        
        self.encoder.eval()
    
    def add_fragrance_property(self, name: str, properties: FragranceProperties) -> None:
        """새로운 향료 속성 추가"""
        # 분자 지문 생성
        if properties.smiles and not properties.molecular_fingerprint:
            properties.molecular_fingerprint = self.chemical_extractor.get_molecular_fingerprint(properties.smiles)
        
        self.fragrance_properties[name] = properties
        
        # 데이터베이스 업데이트 (JSON 형태로 저장하려면 변환 필요)
        self._save_fragrance_properties()
    
    def _save_fragrance_properties(self) -> None:
        """향료 속성 데이터베이스 저장"""
        # 저장을 위해 직렬화 가능한 형태로 변환
        serializable_data = {}
        for name, props in self.fragrance_properties.items():
            serializable_data[name] = {
                'molecular_weight': props.molecular_weight,
                'boiling_point': props.boiling_point,
                'volatility_rate': props.volatility_rate,
                'solubility_water': props.solubility_water,
                'solubility_alcohol': props.solubility_alcohol,
                'odor_intensity': props.odor_intensity,
                'odor_family': props.odor_family,
                'odor_description': props.odor_description,
                'emotional_associations': props.emotional_associations,
                'cultural_associations': props.cultural_associations,
                'preferred_season': props.preferred_season,
                'preferred_time': props.preferred_time,
                'longevity': props.longevity,
                'projection': props.projection,
                'smiles': props.smiles,
                'molecular_fingerprint': props.molecular_fingerprint
            }
        
        try:
            self.fragrance_db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.fragrance_db_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving fragrance properties: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        return {
            'encoder_parameters': sum(p.numel() for p in self.encoder.parameters()),
            'fragrance_count': len(self.fragrance_properties),
            'chemical_extractor_available': self.chemical_extractor.available,
            'image_extractor_available': self.image_extractor.available,
            'device': str(self.device),
            'modalities_supported': ['text', 'chemical', 'image', 'molecular_fingerprint']
        }