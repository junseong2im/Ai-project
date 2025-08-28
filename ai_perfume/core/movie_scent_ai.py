#!/usr/bin/env python3
"""
영화 장면용 고급 냄새 구조 딥러닝 AI 시스템
한계치까지 학습된 다차원 향수 추천 엔진
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)

class MovieScentDataset(Dataset):
    """영화 장면-향수 데이터셋"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, metadata: Optional[np.ndarray] = None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.metadata = torch.FloatTensor(metadata) if metadata is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.metadata is not None:
            return self.features[idx], self.targets[idx], self.metadata[idx]
        return self.features[idx], self.targets[idx]

class AdvancedMovieNeuralNetwork(nn.Module):
    """영화 장면 분석용 고급 신경망
    
    다중 헤드 어텐션, 잔차 연결, 배치 정규화를 포함한 최신 아키텍처
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        super(AdvancedMovieNeuralNetwork, self).__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # 입력 임베딩 층
        self.input_embedding = nn.Linear(input_dim, hidden_dims[0])
        
        # 다중 헤드 어텐션 (선택사항)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[0],
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dims[0])
        
        # 피드포워드 네트워크
        self.layers = nn.ModuleList()
        prev_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ))
            prev_dim = hidden_dim
        
        # 출력 층들
        self.intensity_head = nn.Linear(prev_dim, 1)  # 향수 강도
        self.scent_profile_head = nn.Linear(prev_dim, output_dim - 1)  # 향수 프로필
        
        # 잔차 연결을 위한 프로젝션 층들
        if use_residual:
            self.residual_projections = nn.ModuleList()
            for i, hidden_dim in enumerate(hidden_dims[1:]):
                if i == 0:
                    self.residual_projections.append(nn.Linear(hidden_dims[0], hidden_dim))
                else:
                    self.residual_projections.append(nn.Linear(hidden_dims[i], hidden_dim))
    
    def forward(self, x):
        # 입력 임베딩
        x = self.input_embedding(x)
        
        # 다중 헤드 어텐션
        if self.use_attention:
            # 배치 차원 확장 (시퀀스 길이 1)
            x_seq = x.unsqueeze(1)
            attn_out, _ = self.attention(x_seq, x_seq, x_seq)
            x = self.attention_norm(x + attn_out.squeeze(1))
        
        # 피드포워드 네트워크 (잔차 연결 포함)
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x)
            
            # 잔차 연결
            if self.use_residual and i < len(self.residual_projections):
                residual_projected = self.residual_projections[i](residual)
                x = x + residual_projected
        
        # 출력 헤드들
        intensity = torch.sigmoid(self.intensity_head(x)) * 10  # 0-10 스케일
        scent_profile = torch.sigmoid(self.scent_profile_head(x))
        
        return torch.cat([intensity, scent_profile], dim=1)

class MovieScentProcessor:
    """영화 장면-향수 데이터 전처리기"""
    
    def __init__(self):
        self.movie_data = None
        self.scent_vectorizer = TfidfVectorizer(max_features=500)
        self.emotion_vectorizer = TfidfVectorizer(max_features=200)
        self.visual_vectorizer = TfidfVectorizer(max_features=300)
        self.scalers = {}
        self.encoders = {}
        
        # 향수 노트 카테고리 (확장된 버전)
        self.scent_categories = {
            'citrus': ['bergamot', 'lemon', 'orange', 'grapefruit', 'lime', 'mandarin', 'citron'],
            'floral': ['rose', 'jasmine', 'lily', 'violet', 'tuberose', 'magnolia', 'peony', 'iris'],
            'woody': ['cedar', 'sandalwood', 'pine', 'cypress', 'oak', 'birch', 'bamboo'],
            'oriental': ['amber', 'vanilla', 'musk', 'oud', 'incense', 'benzoin', 'myrrh'],
            'fresh': ['mint', 'eucalyptus', 'sea breeze', 'ozone', 'green leaves', 'cucumber'],
            'spicy': ['cinnamon', 'nutmeg', 'cardamom', 'black pepper', 'clove', 'ginger'],
            'fruity': ['apple', 'peach', 'berry', 'plum', 'apricot', 'pear', 'cherry'],
            'gourmand': ['chocolate', 'coffee', 'caramel', 'honey', 'almond', 'coconut', 'praline'],
            'animalic': ['leather', 'musk', 'ambergris', 'civet', 'castoreum'],
            'herbal': ['basil', 'rosemary', 'thyme', 'sage', 'lavender', 'chamomile'],
            'aquatic': ['ocean', 'rain', 'water lily', 'sea salt', 'marine'],
            'metallic': ['steel', 'iron', 'copper', 'metallic', 'mineral'],
            'smoky': ['smoke', 'tobacco', 'birch tar', 'burnt wood', 'fire'],
            'earthy': ['soil', 'moss', 'mushroom', 'wet earth', 'clay', 'stone'],
            'synthetic': ['aldehydes', 'synthetic', 'chemical', 'laboratory', 'artificial']
        }
        
        logger.info("영화 향수 프로세서 초기화 완료")
    
    def load_movie_data(self, data_path: str) -> Dict:
        """영화 데이터 로드"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.movie_data = json.load(f)
            logger.info(f"영화 데이터 로드 완료: {len(self.movie_data['movie_scenes'])}개 장면")
            return self.movie_data
        except Exception as e:
            logger.error(f"영화 데이터 로드 실패: {e}")
            raise
    
    def expand_dataset(self, base_scenes: List[Dict], multiplier: int = 50) -> List[Dict]:
        """데이터셋 확장을 통한 다양성 증대"""
        expanded_scenes = []
        
        # 기본 장면들 추가
        expanded_scenes.extend(base_scenes)
        
        # 변형된 장면들 생성
        for _ in range(multiplier):
            for base_scene in base_scenes:
                # 장면 변형
                new_scene = self._create_scene_variation(base_scene)
                expanded_scenes.append(new_scene)
        
        logger.info(f"데이터셋 확장 완료: {len(expanded_scenes)}개 장면")
        return expanded_scenes
    
    def _create_scene_variation(self, base_scene: Dict) -> Dict:
        """기본 장면의 변형 생성"""
        variation = base_scene.copy()
        variation['scene_id'] = f"{base_scene['scene_id']}_var_{random.randint(1000, 9999)}"
        
        # 시간대 변형
        times = ['dawn', 'morning', 'afternoon', 'evening', 'night', 'midnight']
        variation['time_of_day'] = random.choice(times)
        
        # 날씨 변형
        weathers = ['sunny', 'cloudy', 'rainy', 'snowy', 'foggy', 'windy', 'stormy']
        variation['weather'] = random.choice(weathers)
        
        # 감정 변형 (기본 감정에 추가)
        additional_emotions = ['melancholy', 'excitement', 'serenity', 'tension', 'euphoria']
        if random.random() < 0.3:
            variation['emotions'].append(random.choice(additional_emotions))
        
        # 향수 프로필 변형
        self._modify_scent_profile(variation['scent_profile'])
        
        return variation
    
    def _modify_scent_profile(self, scent_profile: Dict):
        """향수 프로필 수정"""
        # 강도 변형 (±2 범위)
        scent_profile['intensity'] = max(1, min(10, 
            scent_profile['intensity'] + random.randint(-2, 2)))
        
        # 지속성 변형 (±1 범위)
        scent_profile['longevity'] = max(1, min(10,
            scent_profile['longevity'] + random.randint(-1, 1)))
        
        # 투사력 변형 (±1 범위)
        scent_profile['projection'] = max(1, min(10,
            scent_profile['projection'] + random.randint(-1, 1)))
        
        # 노트 변형 (30% 확률로 노트 추가/제거)
        if random.random() < 0.3:
            # 랜덤 카테고리에서 노트 추가
            category = random.choice(list(self.scent_categories.keys()))
            note = random.choice(self.scent_categories[category])
            
            note_type = random.choice(['primary_notes', 'secondary_notes'])
            if note not in scent_profile[note_type]:
                scent_profile[note_type].append(note)
    
    def extract_features(self, scenes: List[Dict]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """고차원 특성 추출"""
        logger.info("고차원 특성 추출 시작...")
        
        # 기본 특성들
        scene_types = [scene['scene_type'] for scene in scenes]
        locations = [scene['location'] for scene in scenes]
        times = [scene['time_of_day'] for scene in scenes]
        weathers = [scene['weather'] for scene in scenes]
        
        # 텍스트 특성들
        emotions_text = [' '.join(scene['emotions']) for scene in scenes]
        visuals_text = [' '.join(scene['visual_elements']) for scene in scenes]
        scent_notes_text = [' '.join(scene['scent_profile']['primary_notes'] + 
                                   scene['scent_profile']['secondary_notes']) for scene in scenes]
        
        # 벡터화
        emotions_features = self.emotion_vectorizer.fit_transform(emotions_text).toarray()
        visuals_features = self.visual_vectorizer.fit_transform(visuals_text).toarray()
        scent_features = self.scent_vectorizer.fit_transform(scent_notes_text).toarray()
        
        # 카테고리 인코딩
        scene_type_encoder = LabelEncoder()
        location_encoder = LabelEncoder()
        time_encoder = LabelEncoder()
        weather_encoder = LabelEncoder()
        mood_encoder = LabelEncoder()
        
        scene_type_encoded = scene_type_encoder.fit_transform(scene_types)
        location_encoded = location_encoder.fit_transform(locations)
        time_encoded = time_encoder.fit_transform(times)
        weather_encoded = weather_encoder.fit_transform(weathers)
        mood_encoded = mood_encoder.fit_transform([scene['scent_profile']['mood'] for scene in scenes])
        
        # 수치 특성들
        numerical_features = np.array([
            [
                scene['scent_profile']['intensity'],
                scene['scent_profile']['longevity'], 
                scene['scent_profile']['projection'],
                len(scene['emotions']),
                len(scene['visual_elements']),
                len(scene['scent_profile']['primary_notes']),
                len(scene['scent_profile']['secondary_notes'])
            ]
            for scene in scenes
        ])
        
        # 향수 카테고리 특성
        category_features = self._extract_scent_category_features(scenes)
        
        # 시간-날씨 조합 특성
        temporal_features = self._extract_temporal_features(scenes)
        
        # 모든 특성 결합
        all_features = np.concatenate([
            emotions_features,
            visuals_features,
            scene_type_encoded.reshape(-1, 1),
            location_encoded.reshape(-1, 1),
            time_encoded.reshape(-1, 1),
            weather_encoded.reshape(-1, 1),
            mood_encoded.reshape(-1, 1),
            numerical_features,
            category_features,
            temporal_features
        ], axis=1)
        
        # 타겟 생성 (향수 프로필 예측)
        targets = np.array([
            [
                scene['scent_profile']['intensity'] / 10.0,  # 정규화
                scene['scent_profile']['longevity'] / 10.0,
                scene['scent_profile']['projection'] / 10.0,
            ] + self._encode_scent_notes(scene['scent_profile'])
            for scene in scenes
        ])
        
        # 인코더 저장
        self.encoders = {
            'scene_type': scene_type_encoder,
            'location': location_encoder,
            'time': time_encoder,
            'weather': weather_encoder,
            'mood': mood_encoder
        }
        
        logger.info(f"특성 추출 완료: {all_features.shape}, 타겟: {targets.shape}")
        
        return all_features, targets, self.encoders
    
    def _extract_scent_category_features(self, scenes: List[Dict]) -> np.ndarray:
        """향수 카테고리별 특성 추출"""
        category_features = []
        
        for scene in scenes:
            all_notes = (scene['scent_profile']['primary_notes'] + 
                        scene['scent_profile']['secondary_notes'])
            
            # 각 카테고리별 점수 계산
            category_scores = []
            for category, notes in self.scent_categories.items():
                score = sum(1 for note in all_notes if any(n in note.lower() for n in notes))
                category_scores.append(score / max(1, len(all_notes)))  # 정규화
            
            category_features.append(category_scores)
        
        return np.array(category_features)
    
    def _extract_temporal_features(self, scenes: List[Dict]) -> np.ndarray:
        """시간-날씨 조합 특성"""
        temporal_features = []
        
        # 시간대별 가중치
        time_weights = {
            'dawn': [0.8, 0.2, 0.0, 0.0],  # [morning_like, day_like, evening_like, night_like]
            'morning': [1.0, 0.3, 0.0, 0.0],
            'afternoon': [0.0, 1.0, 0.2, 0.0],
            'evening': [0.0, 0.2, 1.0, 0.3],
            'night': [0.0, 0.0, 0.3, 1.0],
            'midnight': [0.0, 0.0, 0.0, 1.0]
        }
        
        # 날씨별 가중치
        weather_weights = {
            'sunny': [1.0, 0.0, 0.0, 0.0],  # [bright, neutral, dark, stormy]
            'cloudy': [0.3, 1.0, 0.2, 0.0],
            'rainy': [0.0, 0.2, 0.8, 0.3],
            'snowy': [0.5, 0.3, 0.5, 0.0],
            'foggy': [0.0, 0.3, 0.7, 0.0],
            'windy': [0.2, 0.8, 0.3, 0.2],
            'stormy': [0.0, 0.0, 0.5, 1.0]
        }
        
        for scene in scenes:
            time_feature = time_weights.get(scene['time_of_day'], [0.25, 0.25, 0.25, 0.25])
            weather_feature = weather_weights.get(scene['weather'], [0.25, 0.25, 0.25, 0.25])
            
            # 시간-날씨 조합
            combined_feature = [t * w for t, w in zip(time_feature, weather_feature)]
            temporal_features.append(time_feature + weather_feature + combined_feature)
        
        return np.array(temporal_features)
    
    def _encode_scent_notes(self, scent_profile: Dict) -> List[float]:
        """향수 노트를 벡터로 인코딩"""
        all_notes = scent_profile['primary_notes'] + scent_profile['secondary_notes']
        
        # 각 카테고리별 강도 계산
        encoded = []
        for category, notes in self.scent_categories.items():
            strength = sum(1 for note in all_notes if any(n in note.lower() for n in notes))
            encoded.append(strength / max(1, len(all_notes)))  # 정규화
        
        return encoded

class MovieScentAI:
    """영화 장면용 최고급 향수 AI 시스템"""
    
    def __init__(self):
        self.processor = MovieScentProcessor()
        self.model = None
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        logger.info("영화 향수 AI 시스템 초기화 완료")
    
    def prepare_ultimate_dataset(self, data_path: str, expansion_factor: int = 100):
        """한계치 데이터셋 준비"""
        logger.info("한계치 데이터셋 준비 시작...")
        
        # 기본 데이터 로드
        movie_data = self.processor.load_movie_data(data_path)
        base_scenes = movie_data['movie_scenes']
        
        # 데이터셋 대대적 확장
        expanded_scenes = self.processor.expand_dataset(base_scenes, expansion_factor)
        
        # 특성 추출
        X, y, encoders = self.processor.extract_features(expanded_scenes)
        
        # 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"최종 데이터셋: {X_scaled.shape[0]}개 샘플, {X_scaled.shape[1]}개 특성")
        
        return X_scaled, y, scaler, encoders
    
    def build_ultimate_model(self, input_dim: int, output_dim: int) -> AdvancedMovieNeuralNetwork:
        """최고급 모델 구축"""
        logger.info("최고급 신경망 모델 구축 중...")
        
        # 최적화된 하이퍼파라미터
        hidden_dims = [1024, 512, 256, 128, 64]
        
        model = AdvancedMovieNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_heads=16,  # 더 많은 어텐션 헤드
            dropout_rate=0.2,  # 낮은 드롭아웃으로 더 많은 정보 보존
            use_attention=True,
            use_residual=True
        )
        
        # 매개변수 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"모델 구축 완료:")
        logger.info(f"  - 총 매개변수: {total_params:,}")
        logger.info(f"  - 훈련 가능 매개변수: {trainable_params:,}")
        
        return model
    
    def train_to_limit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_epochs: int = 1000,
        patience: int = 50,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """한계치까지 훈련"""
        logger.info("한계치 훈련 시작...")
        
        # 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        
        # 데이터셋 생성
        train_dataset = MovieScentDataset(X_train, y_train)
        val_dataset = MovieScentDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 모델 구축
        self.model = self.build_ultimate_model(X.shape[1], y.shape[1])
        
        # 최적화 설정
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=20, factor=0.5, verbose=True
        )
        
        # 손실 함수 (다중 목표)
        mse_loss = nn.MSELoss()
        
        # 훈련
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # 훈련 단계
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_x)
                
                # 다중 손실 계산
                intensity_loss = mse_loss(outputs[:, 0:3], batch_y[:, 0:3])
                profile_loss = mse_loss(outputs[:, 3:], batch_y[:, 3:])
                
                # 가중 총 손실
                total_loss = 0.4 * intensity_loss + 0.6 * profile_loss
                
                total_loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += total_loss.item()
            
            # 검증 단계
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = self.model(batch_x)
                    
                    intensity_loss = mse_loss(outputs[:, 0:3], batch_y[:, 0:3])
                    profile_loss = mse_loss(outputs[:, 3:], batch_y[:, 3:])
                    total_loss = 0.4 * intensity_loss + 0.6 * profile_loss
                    
                    val_loss += total_loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            
            # 학습률 스케줄링
            scheduler.step(avg_val_loss)
            
            # 조기 종료 체크
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # 최고 모델 저장
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_val_loss,
                    'model_config': {
                        'input_dim': X.shape[1],
                        'output_dim': y.shape[1],
                        'hidden_dims': [1024, 512, 256, 128, 64]
                    }
                }, 'models/ultimate_movie_scent_model.pth')
                
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"조기 종료 (에포크 {epoch+1})")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'에포크 [{epoch+1}/{max_epochs}], '
                          f'훈련 손실: {avg_train_loss:.6f}, '
                          f'검증 손실: {avg_val_loss:.6f}')
        
        logger.info(f"훈련 완료! 최고 검증 손실: {best_val_loss:.6f}")
        
        return {
            'best_val_loss': best_val_loss,
            'training_history': self.training_history,
            'total_epochs': epoch + 1
        }
    
    def predict_movie_scent(self, scene_data: Dict) -> Dict[str, Any]:
        """영화 장면용 향수 예측"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다")
        
        # 단일 장면을 리스트로 변환하여 특성 추출
        scene_features, _, _ = self.processor.extract_features([scene_data])
        
        # 예측 수행
        self.model.eval()
        with torch.no_grad():
            scene_tensor = torch.FloatTensor(scene_features)
            prediction = self.model(scene_tensor).squeeze().numpy()
        
        # 예측 결과 해석
        result = {
            'intensity': float(prediction[0] * 10),  # 역정규화
            'longevity': float(prediction[1] * 10),
            'projection': float(prediction[2] * 10),
            'scent_categories': {},
            'confidence': self._calculate_prediction_confidence(prediction),
            'recommended_notes': self._generate_scent_recommendations(prediction[3:])
        }
        
        # 카테고리별 강도
        category_strengths = prediction[3:]
        for i, (category, _) in enumerate(self.processor.scent_categories.items()):
            if i < len(category_strengths):
                result['scent_categories'][category] = float(category_strengths[i])
        
        return result
    
    def _calculate_prediction_confidence(self, prediction: np.ndarray) -> float:
        """예측 신뢰도 계산"""
        # 예측값들의 분산을 기반으로 신뢰도 계산
        variance = np.var(prediction)
        confidence = max(0.1, min(0.95, 1.0 / (1.0 + variance * 10)))
        return confidence
    
    def _generate_scent_recommendations(self, category_strengths: np.ndarray) -> Dict[str, List[str]]:
        """카테고리 강도 기반 향수 노트 추천"""
        recommendations = {'primary_notes': [], 'secondary_notes': []}
        
        # 상위 카테고리들 선택
        category_items = list(self.processor.scent_categories.items())
        top_indices = np.argsort(category_strengths)[-5:][::-1]  # 상위 5개
        
        for i, idx in enumerate(top_indices):
            if idx < len(category_items):
                category, notes = category_items[idx]
                strength = category_strengths[idx]
                
                if strength > 0.3:  # 임계값 이상인 경우만
                    selected_notes = np.random.choice(notes, 
                                                    min(3, len(notes)), 
                                                    replace=False).tolist()
                    
                    if i < 2:  # 상위 2개는 primary
                        recommendations['primary_notes'].extend(selected_notes[:2])
                    else:  # 나머지는 secondary
                        recommendations['secondary_notes'].extend(selected_notes[:1])
        
        return recommendations

def main():
    """메인 실행 함수"""
    try:
        logger.info("🎬 영화용 한계치 냄새 구조 딥러닝 시스템 시작")
        
        # AI 시스템 초기화
        movie_ai = MovieScentAI()
        
        # 한계치 데이터셋 준비
        data_path = "data/movie_scent_database.json"
        X, y, scaler, encoders = movie_ai.prepare_ultimate_dataset(data_path, expansion_factor=200)
        
        logger.info(f"📊 최종 데이터셋 규모:")
        logger.info(f"   - 샘플 수: {X.shape[0]:,}")
        logger.info(f"   - 특성 수: {X.shape[1]:,}")
        logger.info(f"   - 타겟 수: {y.shape[1]:,}")
        
        # 한계치까지 훈련
        training_results = movie_ai.train_to_limit(
            X, y,
            max_epochs=500,
            patience=30,
            batch_size=128
        )
        
        # 결과 출력
        logger.info("🎉 한계치 훈련 완료!")
        logger.info(f"📈 최고 검증 손실: {training_results['best_val_loss']:.6f}")
        logger.info(f"📈 총 에포크: {training_results['total_epochs']}")
        
        # 전처리 도구 저장
        with open('models/movie_scent_preprocessor.pkl', 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'encoders': encoders,
                'processor': movie_ai.processor
            }, f)
        
        # 테스트 예측
        test_scene = {
            "scene_type": "romantic",
            "location": "beach_sunset",
            "time_of_day": "sunset",
            "weather": "warm",
            "emotions": ["love", "passion", "serenity"],
            "visual_elements": ["ocean", "sand", "golden_light"],
            "scent_profile": {
                "primary_notes": ["sea breeze", "white flowers"],
                "secondary_notes": ["vanilla", "amber"],
                "intensity": 6,
                "longevity": 7,
                "projection": 5,
                "mood": "romantic"
            }
        }
        
        prediction = movie_ai.predict_movie_scent(test_scene)
        
        logger.info("🧪 테스트 예측 결과:")
        logger.info(f"   - 강도: {prediction['intensity']:.2f}/10")
        logger.info(f"   - 지속성: {prediction['longevity']:.2f}/10")
        logger.info(f"   - 투사력: {prediction['projection']:.2f}/10")
        logger.info(f"   - 신뢰도: {prediction['confidence']:.3f}")
        
        top_categories = sorted(prediction['scent_categories'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        logger.info("   - 상위 향수 카테고리:")
        for category, strength in top_categories:
            logger.info(f"     * {category}: {strength:.3f}")
        
        logger.info("✅ 영화용 냄새 구조 딥러닝 시스템 완성!")
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()