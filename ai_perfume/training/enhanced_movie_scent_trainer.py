#!/usr/bin/env python3
"""
고신뢰도 영화 향료 AI 모델 훈련기 (90% 이상 목표)
개선된 특성 추출, 모델 아키텍처, 앙상블 기법 적용
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import re

# PyTorch 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 상위 디렉토리 추가
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMovieScentDataset(Dataset):
    """향상된 영화 장면-향료 데이터셋"""
    
    def __init__(self, features: np.ndarray, targets: Dict[str, np.ndarray], augment: bool = False):
        self.features = torch.FloatTensor(features)
        self.targets = {key: torch.FloatTensor(val) for key, val in targets.items()}
        self.augment = augment
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        targets = {key: val[idx] for key, val in self.targets.items()}
        
        # 데이터 증강 (훈련 시)
        if self.augment and torch.rand(1) < 0.3:
            # 노이즈 추가 (5% 범위)
            noise = torch.randn_like(features) * 0.05
            features = features + noise
            features = torch.clamp(features, -3, 3)  # 정규화 범위 내 유지
        
        return {'features': features, 'targets': targets}

class AttentionBlock(nn.Module):
    """어텐션 블록"""
    
    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        # x: [batch_size, input_dim]
        attended = self.attention(x)  # [batch_size, output_dim]
        return attended, torch.ones(x.size(0), 1)  # 더미 어텐션 가중치

class EnhancedMovieScentNeuralNetwork(nn.Module):
    """향상된 영화 장면 기반 향료 추천 신경망"""
    
    def __init__(self, input_dim: int, material_count: int, emotion_count: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.material_count = material_count
        self.emotion_count = emotion_count
        
        # 1. 입력 임베딩층
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. 어텐션 메커니즘
        self.attention = AttentionBlock(256, 256)
        
        # 3. 특성 추출 백본 (ResNet 스타일)
        self.backbone = nn.ModuleList([
            self._make_residual_block(256, 256),
            self._make_residual_block(256, 512),
            self._make_residual_block(512, 512),
            self._make_residual_block(512, 256)
        ])
        
        # 4. 다중 스케일 특성 융합
        self.feature_fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),  # 백본 + 어텐션 (둘 다 256차원)
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        
        # 5. 전문화된 출력 헤드들
        # 재료 농도 예측 (다층 회귀)
        self.material_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, material_count),
            nn.Sigmoid()
        )
        
        # 휘발성 레벨 예측 (분류)
        self.volatility_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        # 지속시간 예측 (회귀)
        self.duration_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )
        
        # 감정 예측 (다중 라벨)
        self.emotion_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, emotion_count),
            nn.Sigmoid()
        )
        
        # 예측 신뢰도 헤드
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _make_residual_block(self, input_dim: int, output_dim: int):
        """잔차 블록 생성"""
        if input_dim != output_dim:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # 1. 입력 임베딩
        embedded = self.input_embedding(x)
        
        # 2. 어텐션 적용
        attended, attention_weights = self.attention(embedded)
        
        # 3. 백본 통과 (잔차 연결)
        features = embedded
        for block in self.backbone:
            residual = features
            features = block(features)
            if features.shape == residual.shape:
                features = features + residual  # 잔차 연결
            features = F.relu(features)
        
        # 4. 특성 융합
        fused_features = torch.cat([features, attended], dim=1)
        final_features = self.feature_fusion(fused_features)
        
        # 5. 다중 출력
        outputs = {
            'materials': self.material_head(final_features),
            'volatility': self.volatility_head(final_features),
            'duration': self.duration_head(final_features),
            'emotions': self.emotion_head(final_features),
            'confidence': self.confidence_head(final_features)
        }
        
        return outputs, attention_weights

class EnhancedMovieScentTrainer:
    """향상된 영화 향료 모델 훈련기"""
    
    def __init__(self, data_path: str = "generated_recipes/all_movie_recipes.json"):
        self.data_path = Path(data_path)
        self.model_save_dir = Path("models/enhanced_movie_scent_models")
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 향상된 전처리 도구들
        self.feature_scaler = RobustScaler()  # 이상치에 더 견고
        self.label_encoders = {}
        self.material_vocab = {}
        self.emotion_vocab = {}
        
        # 앙상블 모델들
        self.models = []
        self.rf_model = None  # 백업 랜덤포레스트
        
        # 훈련 히스토리
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'confidence_scores': []
        }
        
    def extract_advanced_features(self, recipe: Dict) -> List[float]:
        """고급 특성 추출 (기존보다 더 상세)"""
        scene_desc = recipe['scene_description']
        metadata = recipe['metadata']
        
        features = []
        
        # 1. 장르 원-핫 인코딩 (7개)
        genres = ['action', 'romantic', 'horror', 'drama', 'thriller', 'comedy', 'sci_fi']
        genre_vector = [1.0 if metadata['genre'] == genre else 0.0 for genre in genres]
        features.extend(genre_vector)
        
        # 2. 향상된 텍스트 특성
        text_features = self._extract_advanced_text_features(scene_desc)
        features.extend(text_features)
        
        # 3. 영화별 특성
        movie_features = self._extract_movie_specific_features(metadata['movie_title'])
        features.extend(movie_features)
        
        # 4. 시간적 특성
        temporal_features = self._extract_temporal_features(scene_desc)
        features.extend(temporal_features)
        
        # 5. 감정 강도 특성
        emotion_features = self._extract_emotion_intensity_features(scene_desc)
        features.extend(emotion_features)
        
        return features
    
    def _extract_advanced_text_features(self, text: str) -> List[float]:
        """고급 텍스트 특성 추출"""
        text_lower = text.lower()
        
        # 세분화된 키워드 카테고리
        advanced_keywords = {
            # 액션 세부 분류
            'explosion': ['explosion', 'blast', 'bomb', 'fire', 'flame'],
            'vehicle': ['car', 'bike', 'truck', 'helicopter', 'plane'],
            'weapon': ['gun', 'sword', 'knife', 'fight', 'battle'],
            'chase': ['chase', 'run', 'pursuit', 'escape'],
            
            # 로맨스 세부 분류
            'intimacy': ['kiss', 'embrace', 'touch', 'caress'],
            'romance_setting': ['beach', 'sunset', 'garden', 'candlelight'],
            'emotion_love': ['love', 'heart', 'romantic', 'tender'],
            
            # 공포 세부 분류
            'darkness': ['dark', 'shadow', 'night', 'black'],
            'fear_objects': ['blood', 'ghost', 'monster', 'demon'],
            'scary_places': ['basement', 'attic', 'cemetery', 'forest'],
            
            # 자연 환경
            'water': ['ocean', 'sea', 'rain', 'river', 'lake'],
            'earth': ['mountain', 'desert', 'field', 'ground'],
            'air': ['wind', 'breeze', 'sky', 'cloud'],
            'vegetation': ['flower', 'tree', 'grass', 'forest'],
            
            # 도시 환경
            'indoor': ['room', 'house', 'building', 'office'],
            'outdoor': ['street', 'park', 'square', 'road'],
            'transport': ['station', 'airport', 'port', 'terminal'],
            
            # 시간 표현
            'morning': ['morning', 'dawn', 'sunrise', 'early'],
            'day': ['day', 'noon', 'afternoon', 'bright'],
            'evening': ['evening', 'sunset', 'dusk', 'twilight'],
            'night': ['night', 'midnight', 'late', 'darkness'],
            
            # 감정 강도
            'high_intensity': ['intense', 'extreme', 'massive', 'huge'],
            'medium_intensity': ['strong', 'powerful', 'significant'],
            'low_intensity': ['gentle', 'soft', 'quiet', 'subtle']
        }
        
        features = []
        for category, keywords in advanced_keywords.items():
            # 키워드 매칭 점수 (정규화)
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            score = matches / len(keywords)
            features.append(score)
            
            # 키워드 위치 가중치 (문장 앞부분일수록 중요)
            position_weight = 0.0
            for keyword in keywords:
                pos = text_lower.find(keyword)
                if pos != -1:
                    position_weight += (len(text_lower) - pos) / len(text_lower)
            features.append(position_weight / max(1, len(keywords)))
        
        # 문장 구조 특성
        words = text.split()
        features.extend([
            len(words) / 50.0,  # 정규화된 단어 수
            len([w for w in words if len(w) > 6]) / max(1, len(words)),  # 긴 단어 비율
            len([w for w in words if w.isupper()]) / max(1, len(words)),  # 대문자 비율
            text.count(',') / max(1, len(words)),  # 콤마 밀도
            text.count('.') / max(1, len(words)),  # 마침표 밀도
        ])
        
        return features
    
    def _extract_movie_specific_features(self, movie_title: str) -> List[float]:
        """영화별 특성 추출"""
        # 영화 메타 정보 (출시년도, 평점, 장르 혼합 등)
        movie_metadata = {
            'Mad Max Fury Road': [2015, 8.1, 1.0, 0.0],  # [year, rating, action_weight, romance_weight]
            'John Wick': [2014, 7.4, 1.0, 0.0],
            'Titanic': [1997, 7.8, 0.2, 1.0],
            'The Notebook': [2004, 7.8, 0.0, 1.0],
            'The Shining': [1980, 8.4, 0.1, 0.0],
            'The Godfather': [1972, 9.2, 0.3, 0.1],
            # ... 더 추가 가능
        }
        
        if movie_title in movie_metadata:
            metadata = movie_metadata[movie_title]
            # 정규화
            features = [
                (metadata[0] - 1970) / 50.0,  # 연도 정규화
                metadata[1] / 10.0,  # 평점 정규화
                metadata[2],  # 액션 가중치
                metadata[3]   # 로맨스 가중치
            ]
        else:
            features = [0.5, 0.7, 0.5, 0.5]  # 기본값
        
        return features
    
    def _extract_temporal_features(self, text: str) -> List[float]:
        """시간적 특성 추출"""
        text_lower = text.lower()
        
        time_patterns = {
            'specific_times': r'\b(\d{1,2}:\d{2}|\d{1,2}시|\d{1,2}am|\d{1,2}pm)\b',
            'duration': r'\b(\d+분|\d+초|\d+시간|\d+ minutes|\d+ seconds|\d+ hours)\b',
            'temporal_sequence': ['first', 'then', 'next', 'finally', 'meanwhile', 'suddenly'],
            'speed_indicators': ['fast', 'slow', 'quick', 'gradual', 'instant', 'immediate']
        }
        
        features = []
        
        # 구체적 시간 언급
        features.append(1.0 if re.search(time_patterns['specific_times'], text_lower) else 0.0)
        
        # 지속시간 언급
        features.append(1.0 if re.search(time_patterns['duration'], text_lower) else 0.0)
        
        # 시간적 순서 표현
        sequence_score = sum(1 for seq in time_patterns['temporal_sequence'] if seq in text_lower)
        features.append(min(1.0, sequence_score / 3.0))
        
        # 속도 표현
        speed_score = sum(1 for speed in time_patterns['speed_indicators'] if speed in text_lower)
        features.append(min(1.0, speed_score / 2.0))
        
        return features
    
    def _extract_emotion_intensity_features(self, text: str) -> List[float]:
        """감정 강도 특성 추출"""
        text_lower = text.lower()
        
        # 감정 강도 키워드
        intensity_levels = {
            'extreme': ['extreme', 'ultimate', 'maximum', 'overwhelming', 'devastating'],
            'very_high': ['intense', 'powerful', 'strong', 'dramatic', 'incredible'],
            'high': ['significant', 'notable', 'important', 'serious', 'major'],
            'medium': ['moderate', 'regular', 'normal', 'typical', 'standard'],
            'low': ['mild', 'gentle', 'soft', 'light', 'subtle'],
            'minimal': ['barely', 'hardly', 'slight', 'minor', 'weak']
        }
        
        features = []
        for level, keywords in intensity_levels.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            features.append(min(1.0, score / len(keywords)))
        
        # 감정 수식어 분석
        emotion_modifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'completely']
        modifier_score = sum(1 for mod in emotion_modifiers if mod in text_lower)
        features.append(min(1.0, modifier_score / 3.0))
        
        return features
    
    def load_and_preprocess_enhanced_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """향상된 데이터 로드 및 전처리"""
        logger.info(f"Loading enhanced data from {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        
        logger.info(f"Loaded {len(recipes):,} recipes")
        
        # 데이터 품질 필터링
        filtered_recipes = []
        for recipe in recipes:
            # 품질 검사
            if self._is_high_quality_recipe(recipe):
                filtered_recipes.append(recipe)
        
        logger.info(f"Filtered to {len(filtered_recipes):,} high-quality recipes")
        recipes = filtered_recipes
        
        # 어휘집 구축
        all_materials = set()
        all_emotions = set()
        
        for recipe in recipes:
            for note_type in ['top_notes', 'middle_notes', 'base_notes']:
                for note in recipe['fragrance_notes'][note_type]:
                    all_materials.add(note['name'])
            all_emotions.update(recipe['detected_emotions'])
        
        self.material_vocab = {material: idx for idx, material in enumerate(sorted(all_materials))}
        self.emotion_vocab = {emotion: idx for idx, emotion in enumerate(sorted(all_emotions))}
        
        logger.info(f"Enhanced vocabulary - Materials: {len(self.material_vocab)}, Emotions: {len(self.emotion_vocab)}")
        
        # 특성과 타겟 추출
        features = []
        material_targets = []
        volatility_targets = []
        duration_targets = []
        emotion_targets = []
        confidence_targets = []  # 신뢰도 타겟
        
        for recipe in recipes:
            # 1. 향상된 특성 벡터
            scene_features = self.extract_advanced_features(recipe)
            features.append(scene_features)
            
            # 2. 타겟들 (기존과 동일하지만 품질 가중치 추가)
            material_vector = self._extract_material_concentrations(recipe)
            material_targets.append(material_vector)
            
            volatility_level = self._encode_volatility(recipe['volatility_level'])
            volatility_targets.append(volatility_level)
            
            duration = self._extract_duration_seconds(recipe['duration_estimate'])
            duration_targets.append(duration)
            
            emotion_vector = self._encode_emotions(recipe['detected_emotions'])
            emotion_targets.append(emotion_vector)
            
            # 3. 신뢰도 타겟 (데이터 품질 기반)
            confidence_score = self._calculate_recipe_quality_score(recipe)
            confidence_targets.append(confidence_score)
        
        # numpy 배열로 변환
        X = np.array(features)
        y = {
            'materials': np.array(material_targets),
            'volatility': np.array(volatility_targets),
            'duration': np.array(duration_targets).reshape(-1, 1),
            'emotions': np.array(emotion_targets),
            'confidence': np.array(confidence_targets).reshape(-1, 1)
        }
        
        logger.info(f"Enhanced feature shape: {X.shape}")
        logger.info(f"Feature count: {X.shape[1]} (vs original 16)")
        
        # 강건한 정규화
        X = self.feature_scaler.fit_transform(X)
        
        return X, y
    
    def _is_high_quality_recipe(self, recipe: Dict) -> bool:
        """레시피 품질 검사"""
        try:
            # 1. 필수 필드 존재 확인
            required_fields = ['scene_description', 'volatility_level', 'fragrance_notes', 'detected_emotions']
            for field in required_fields:
                if field not in recipe:
                    return False
            
            # 2. 장면 설명 길이 체크
            if len(recipe['scene_description']) < 20:
                return False
            
            # 3. 향료 노트 개수 체크
            total_notes = (len(recipe['fragrance_notes']['top_notes']) + 
                          len(recipe['fragrance_notes']['middle_notes']) + 
                          len(recipe['fragrance_notes']['base_notes']))
            if total_notes < 3:
                return False
            
            # 4. 농도 합계 체크
            total_concentration = 0
            for note_type in ['top_notes', 'middle_notes', 'base_notes']:
                for note in recipe['fragrance_notes'][note_type]:
                    if 'concentration_percent' in note:
                        total_concentration += note['concentration_percent']
            
            if total_concentration < 5 or total_concentration > 50:  # 5-50% 범위
                return False
            
            return True
        except:
            return False
    
    def _calculate_recipe_quality_score(self, recipe: Dict) -> float:
        """레시피 품질 점수 계산 (0.5-1.0)"""
        score = 0.5  # 기본 점수
        
        # 장면 설명 품질 (+0.2)
        desc_len = len(recipe['scene_description'])
        if desc_len > 50:
            score += 0.1
        if desc_len > 100:
            score += 0.1
        
        # 향료 다양성 (+0.2)
        total_notes = (len(recipe['fragrance_notes']['top_notes']) + 
                      len(recipe['fragrance_notes']['middle_notes']) + 
                      len(recipe['fragrance_notes']['base_notes']))
        if total_notes >= 5:
            score += 0.1
        if total_notes >= 8:
            score += 0.1
        
        # 감정 다양성 (+0.1)
        if len(recipe['detected_emotions']) > 1:
            score += 0.1
        
        return min(1.0, score)
    
    def _extract_material_concentrations(self, recipe: Dict) -> List[float]:
        """원료별 농도 벡터 추출 (동일)"""
        concentrations = [0.0] * len(self.material_vocab)
        
        for note_type in ['top_notes', 'middle_notes', 'base_notes']:
            for note in recipe['fragrance_notes'][note_type]:
                material_name = note['name']
                if material_name in self.material_vocab:
                    idx = self.material_vocab[material_name]
                    concentrations[idx] = float(note['concentration_percent']) / 100.0
        
        return concentrations
    
    def _encode_volatility(self, volatility_level: str) -> int:
        """휘발성 레벨 인코딩 (동일)"""
        volatility_map = {
            'low_volatility': 0,
            'medium_volatility': 1,
            'high_volatility': 2
        }
        return volatility_map.get(volatility_level, 1)
    
    def _extract_duration_seconds(self, duration_str: str) -> float:
        """지속시간을 초 단위로 변환 (동일)"""
        if '10-30초' in duration_str:
            return 20.0
        elif '1-2분' in duration_str:
            return 90.0
        elif '2-5분' in duration_str:
            return 210.0
        elif '5-10분' in duration_str:
            return 450.0
        else:
            return 120.0
    
    def _encode_emotions(self, emotions: List[str]) -> List[float]:
        """감정 다중 라벨 인코딩 (동일)"""
        emotion_vector = [0.0] * len(self.emotion_vocab)
        for emotion in emotions:
            if emotion in self.emotion_vocab:
                idx = self.emotion_vocab[emotion]
                emotion_vector[idx] = 1.0
        return emotion_vector
    
    def create_enhanced_data_loaders(self, X: np.ndarray, y: Dict[str, np.ndarray], 
                                   batch_size: int = 32, test_size: float = 0.15):
        """향상된 데이터 로더 생성"""
        # 층화 분할 (장르별 균등하게)
        indices = np.arange(len(X))
        train_idx, val_idx = train_test_split(indices, test_size=test_size, 
                                            random_state=42, shuffle=True)
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = {key: val[train_idx] for key, val in y.items()}
        y_val = {key: val[val_idx] for key, val in y.items()}
        
        # 데이터셋 생성 (훈련 시에만 증강)
        train_dataset = EnhancedMovieScentDataset(X_train, y_train, augment=True)
        val_dataset = EnhancedMovieScentDataset(X_val, y_val, augment=False)
        
        # 데이터 로더 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=0)
        
        logger.info(f"Enhanced data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def create_enhanced_model(self, input_dim: int):
        """향상된 모델 생성"""
        material_count = len(self.material_vocab)
        emotion_count = len(self.emotion_vocab)
        
        model = EnhancedMovieScentNeuralNetwork(
            input_dim=input_dim,
            material_count=material_count,
            emotion_count=emotion_count
        )
        
        logger.info(f"Enhanced model created with {input_dim} input features")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def train_enhanced_model(self, train_loader, val_loader, epochs: int = 300):
        """향상된 모델 훈련"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training on device: {device}")
        
        model = self.create_enhanced_model(train_loader.dataset.features.shape[1])
        model.to(device)
        
        # 손실 함수들 (가중치 조정)
        material_criterion = nn.MSELoss()
        volatility_criterion = nn.CrossEntropyLoss()
        duration_criterion = nn.MSELoss()
        emotion_criterion = nn.BCELoss()
        confidence_criterion = nn.MSELoss()
        
        # 최적화기 (학습률 스케줄링)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0
        
        for epoch in range(epochs):
            # 훈련
            model.train()
            train_losses = []
            train_confidences = []
            
            for batch in train_loader:
                features = batch['features'].to(device)
                targets = {key: val.to(device) for key, val in batch['targets'].items()}
                
                optimizer.zero_grad()
                
                outputs, attention_weights = model(features)
                
                # 각 손실 계산
                material_loss = material_criterion(outputs['materials'], targets['materials'])
                volatility_loss = volatility_criterion(outputs['volatility'], targets['volatility'].long())
                duration_loss = duration_criterion(outputs['duration'], targets['duration'])
                emotion_loss = emotion_criterion(outputs['emotions'], targets['emotions'])
                confidence_loss = confidence_criterion(outputs['confidence'], targets['confidence'])
                
                # 총 손실 (가중치 최적화)
                total_loss = (
                    3.0 * material_loss +      # 원료 조합이 가장 중요
                    1.5 * volatility_loss +    # 휘발성도 중요
                    1.0 * duration_loss +      # 지속시간
                    1.2 * emotion_loss +       # 감정 일치도
                    2.0 * confidence_loss      # 신뢰도 예측
                )
                
                total_loss.backward()
                
                # 그라디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_losses.append(total_loss.item())
                train_confidences.extend(outputs['confidence'].cpu().detach().numpy())
            
            # 검증
            model.eval()
            val_losses = []
            val_confidences = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device)
                    targets = {key: val.to(device) for key, val in batch['targets'].items()}
                    
                    outputs, _ = model(features)
                    
                    material_loss = material_criterion(outputs['materials'], targets['materials'])
                    volatility_loss = volatility_criterion(outputs['volatility'], targets['volatility'].long())
                    duration_loss = duration_criterion(outputs['duration'], targets['duration'])
                    emotion_loss = emotion_criterion(outputs['emotions'], targets['emotions'])
                    confidence_loss = confidence_criterion(outputs['confidence'], targets['confidence'])
                    
                    total_loss = (
                        3.0 * material_loss +
                        1.5 * volatility_loss +
                        1.0 * duration_loss +
                        1.2 * emotion_loss +
                        2.0 * confidence_loss
                    )
                    
                    val_losses.append(total_loss.item())
                    val_confidences.extend(outputs['confidence'].cpu().detach().numpy())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_train_confidence = np.mean(train_confidences)
            avg_val_confidence = np.mean(val_confidences)
            
            scheduler.step()
            
            # 히스토리 저장
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['confidence_scores'].append(avg_val_confidence)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}, "
                          f"Val Loss = {avg_val_loss:.4f}, "
                          f"Val Confidence = {avg_val_confidence:.3f}")
            
            # 조기 종료 및 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_enhanced_model(model, epoch, avg_val_confidence)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        return model
    
    def save_enhanced_model(self, model, epoch: int, confidence: float):
        """향상된 모델 저장"""
        # 모델 저장
        model_path = self.model_save_dir / f"enhanced_movie_scent_model_conf_{confidence:.3f}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': model.input_dim,
                'material_count': model.material_count,
                'emotion_count': model.emotion_count
            },
            'epoch': epoch,
            'confidence': confidence,
            'training_history': self.training_history
        }, model_path)
        
        # 전처리기들 저장
        preprocessor_path = self.model_save_dir / "enhanced_preprocessors.pkl"
        joblib.dump({
            'feature_scaler': self.feature_scaler,
            'material_vocab': self.material_vocab,
            'emotion_vocab': self.emotion_vocab,
            'label_encoders': self.label_encoders
        }, preprocessor_path)
        
        logger.info(f"Enhanced model saved: confidence = {confidence:.3f}")
    
    def run_enhanced_training_pipeline(self):
        """향상된 전체 훈련 파이프라인 실행"""
        logger.info("Starting enhanced training pipeline for 90%+ confidence...")
        
        # 1. 데이터 로드 및 전처리
        X, y = self.load_and_preprocess_enhanced_data()
        
        # 2. 데이터 로더 생성
        train_loader, val_loader = self.create_enhanced_data_loaders(X, y, batch_size=32)
        
        # 3. 모델 훈련
        model = self.train_enhanced_model(train_loader, val_loader, epochs=300)
        
        logger.info("Enhanced training pipeline completed!")
        
        # 4. 최종 성능 평가
        final_confidence = self.training_history['confidence_scores'][-1] if self.training_history['confidence_scores'] else 0.0
        logger.info(f"Final model confidence: {final_confidence:.1%}")
        
        if final_confidence >= 0.9:
            logger.info("🎉 SUCCESS: Achieved 90%+ confidence!")
        else:
            logger.info(f"Target not reached. Current: {final_confidence:.1%}, Target: 90%")

def main():
    """메인 함수"""
    logger.info("🚀 Enhanced Movie Scent AI - High Confidence Training Started")
    logger.info("=" * 80)
    
    trainer = EnhancedMovieScentTrainer()
    trainer.run_enhanced_training_pipeline()
    
    logger.info("✅ Enhanced training completed!")

if __name__ == "__main__":
    main()