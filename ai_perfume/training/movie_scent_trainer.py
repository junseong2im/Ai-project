#!/usr/bin/env python3
"""
영화 장면 → 향료 조합 딥러닝 모델 훈련기
10만개 데이터셋으로 정확한 원료 조합 학습
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# PyTorch 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 상위 디렉토리 추가
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieScentDataset(Dataset):
    """영화 장면-향료 데이터셋"""
    
    def __init__(self, features: np.ndarray, targets: Dict[str, np.ndarray]):
        self.features = torch.FloatTensor(features)
        self.targets = {key: torch.FloatTensor(val) for key, val in targets.items()}
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'targets': {key: val[idx] for key, val in self.targets.items()}
        }

class MovieScentNeuralNetwork(nn.Module):
    """영화 장면 기반 향료 추천 신경망"""
    
    def __init__(self, input_dim: int, material_count: int, genre_count: int, emotion_count: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.material_count = material_count
        self.genre_count = genre_count
        self.emotion_count = emotion_count
        
        # 공통 특성 추출층
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1)
        )
        
        # 다중 출력 헤드들
        # 1. 원료 농도 예측 (회귀)
        self.material_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, material_count),
            nn.Sigmoid()  # 0-1 사이 농도
        )
        
        # 2. 휘발성 레벨 예측 (분류)
        self.volatility_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # high, medium, low
        )
        
        # 3. 지속시간 예측 (회귀)
        self.duration_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # 양수 시간
        )
        
        # 4. 감정 분류
        self.emotion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, emotion_count),
            nn.Sigmoid()  # 다중 감정 가능
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        return {
            'materials': self.material_head(features),
            'volatility': self.volatility_head(features),
            'duration': self.duration_head(features),
            'emotions': self.emotion_head(features)
        }

class MovieScentTrainer:
    """영화 향료 모델 훈련기"""
    
    def __init__(self, data_path: str = "generated_recipes/all_movie_recipes.json"):
        self.data_path = Path(data_path)
        self.model_save_dir = Path("models/movie_scent_models")
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 전처리 도구들
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        self.material_vocab = {}
        self.emotion_vocab = {}
        
        # 모델과 데이터
        self.model = None
        self.train_loader = None
        self.val_loader = None
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """데이터 로드 및 전처리"""
        logger.info(f"Loading data from {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        
        logger.info(f"Loaded {len(recipes):,} recipes")
        
        # 특성과 타겟 추출
        features = []
        material_targets = []
        volatility_targets = []
        duration_targets = []
        emotion_targets = []
        
        # 재료 어휘집 구축
        all_materials = set()
        all_emotions = set()
        
        for recipe in recipes:
            # 모든 재료와 감정 수집
            for note_type in ['top_notes', 'middle_notes', 'base_notes']:
                for note in recipe['fragrance_notes'][note_type]:
                    all_materials.add(note['name'])
            all_emotions.update(recipe['detected_emotions'])
        
        self.material_vocab = {material: idx for idx, material in enumerate(sorted(all_materials))}
        self.emotion_vocab = {emotion: idx for idx, emotion in enumerate(sorted(all_emotions))}
        
        logger.info(f"Material vocabulary size: {len(self.material_vocab)}")
        logger.info(f"Emotion vocabulary size: {len(self.emotion_vocab)}")
        
        # 각 레시피 처리
        for recipe in recipes:
            # 1. 특성 벡터 생성 (장면 설명 기반)
            scene_features = self._extract_scene_features(recipe)
            features.append(scene_features)
            
            # 2. 재료 농도 벡터
            material_vector = self._extract_material_concentrations(recipe)
            material_targets.append(material_vector)
            
            # 3. 휘발성 레벨
            volatility_level = self._encode_volatility(recipe['volatility_level'])
            volatility_targets.append(volatility_level)
            
            # 4. 지속시간 (초 단위로 변환)
            duration = self._extract_duration_seconds(recipe['duration_estimate'])
            duration_targets.append(duration)
            
            # 5. 감정 벡터
            emotion_vector = self._encode_emotions(recipe['detected_emotions'])
            emotion_targets.append(emotion_vector)
        
        # numpy 배열로 변환
        X = np.array(features)
        y = {
            'materials': np.array(material_targets),
            'volatility': np.array(volatility_targets),
            'duration': np.array(duration_targets).reshape(-1, 1),
            'emotions': np.array(emotion_targets)
        }
        
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Material target shape: {y['materials'].shape}")
        logger.info(f"Volatility target shape: {y['volatility'].shape}")
        logger.info(f"Duration target shape: {y['duration'].shape}")
        logger.info(f"Emotion target shape: {y['emotions'].shape}")
        
        # 특성 정규화
        X = self.feature_scaler.fit_transform(X)
        
        return X, y
    
    def _extract_scene_features(self, recipe: Dict) -> List[float]:
        """장면으로부터 특성 벡터 추출"""
        scene_desc = recipe['scene_description']
        metadata = recipe['metadata']
        
        features = []
        
        # 1. 장르 원-핫 인코딩 (7개 장르)
        genres = ['action', 'romantic', 'horror', 'drama', 'thriller', 'comedy', 'sci_fi']
        genre_vector = [1.0 if metadata['genre'] == genre else 0.0 for genre in genres]
        features.extend(genre_vector)
        
        # 2. 텍스트 특성 (키워드 기반)
        text_features = self._extract_text_features(scene_desc)
        features.extend(text_features)
        
        # 3. 메타데이터 특성
        features.append(float(metadata.get('recipe_id', 0)) / 100000.0)  # 정규화
        
        return features
    
    def _extract_text_features(self, text: str) -> List[float]:
        """텍스트에서 특성 추출"""
        text_lower = text.lower()
        
        # 감정 키워드
        emotion_keywords = {
            'action': ['explosion', 'chase', 'fight', 'battle', 'war', 'gun', 'car'],
            'romantic': ['love', 'kiss', 'couple', 'romantic', 'heart', 'tender'],
            'scary': ['dark', 'scary', 'horror', 'fear', 'blood', 'nightmare'],
            'peaceful': ['calm', 'quiet', 'peaceful', 'serene', 'meditation'],
            'nature': ['forest', 'ocean', 'mountain', 'garden', 'flower', 'tree'],
            'urban': ['city', 'street', 'building', 'office', 'hotel', 'club'],
            'temporal': ['morning', 'night', 'sunset', 'dawn', 'evening', 'midnight']
        }
        
        features = []
        for category, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
            features.append(score)
        
        # 텍스트 길이 특성
        features.append(len(text.split()) / 20.0)  # 정규화된 단어 수
        
        return features
    
    def _extract_material_concentrations(self, recipe: Dict) -> List[float]:
        """원료별 농도 벡터 추출"""
        concentrations = [0.0] * len(self.material_vocab)
        
        for note_type in ['top_notes', 'middle_notes', 'base_notes']:
            for note in recipe['fragrance_notes'][note_type]:
                material_name = note['name']
                if material_name in self.material_vocab:
                    idx = self.material_vocab[material_name]
                    concentrations[idx] = float(note['concentration_percent']) / 100.0  # 0-1 정규화
        
        return concentrations
    
    def _encode_volatility(self, volatility_level: str) -> int:
        """휘발성 레벨 인코딩"""
        volatility_map = {
            'low_volatility': 0,
            'medium_volatility': 1,
            'high_volatility': 2
        }
        return volatility_map.get(volatility_level, 1)
    
    def _extract_duration_seconds(self, duration_str: str) -> float:
        """지속시간을 초 단위로 변환"""
        if '10-30초' in duration_str:
            return 20.0
        elif '1-2분' in duration_str:
            return 90.0
        elif '2-5분' in duration_str:
            return 210.0
        elif '5-10분' in duration_str:
            return 450.0
        else:
            return 120.0  # 기본값
    
    def _encode_emotions(self, emotions: List[str]) -> List[float]:
        """감정 다중 라벨 인코딩"""
        emotion_vector = [0.0] * len(self.emotion_vocab)
        for emotion in emotions:
            if emotion in self.emotion_vocab:
                idx = self.emotion_vocab[emotion]
                emotion_vector[idx] = 1.0
        return emotion_vector
    
    def create_data_loaders(self, X: np.ndarray, y: Dict[str, np.ndarray], 
                           batch_size: int = 64, test_size: float = 0.2):
        """데이터 로더 생성"""
        # 훈련/검증 분할
        indices = np.arange(len(X))
        train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = {key: val[train_idx] for key, val in y.items()}
        y_val = {key: val[val_idx] for key, val in y.items()}
        
        # 데이터셋 생성
        train_dataset = MovieScentDataset(X_train, y_train)
        val_dataset = MovieScentDataset(X_val, y_val)
        
        # 데이터 로더 생성
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    def create_model(self, input_dim: int):
        """모델 생성"""
        material_count = len(self.material_vocab)
        emotion_count = len(self.emotion_vocab)
        
        self.model = MovieScentNeuralNetwork(
            input_dim=input_dim,
            material_count=material_count,
            genre_count=7,
            emotion_count=emotion_count
        )
        
        logger.info(f"Model created with input_dim={input_dim}")
        logger.info(f"Material count: {material_count}")
        logger.info(f"Emotion count: {emotion_count}")
    
    def train_model(self, epochs: int = 100, lr: float = 0.001):
        """모델 훈련"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        self.model.to(device)
        
        # 손실 함수들
        material_criterion = nn.MSELoss()  # 농도 회귀
        volatility_criterion = nn.CrossEntropyLoss()  # 휘발성 분류
        duration_criterion = nn.MSELoss()  # 지속시간 회귀
        emotion_criterion = nn.BCELoss()  # 감정 다중 라벨
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 훈련
            self.model.train()
            train_losses = []
            
            for batch in self.train_loader:
                features = batch['features'].to(device)
                targets = {key: val.to(device) for key, val in batch['targets'].items()}
                
                optimizer.zero_grad()
                
                outputs = self.model(features)
                
                # 각 손실 계산
                material_loss = material_criterion(outputs['materials'], targets['materials'])
                volatility_loss = volatility_criterion(outputs['volatility'], targets['volatility'].long())
                duration_loss = duration_criterion(outputs['duration'], targets['duration'])
                emotion_loss = emotion_criterion(outputs['emotions'], targets['emotions'])
                
                # 총 손실 (가중 합)
                total_loss = (
                    2.0 * material_loss +    # 원료 조합이 가장 중요
                    1.0 * volatility_loss +
                    0.5 * duration_loss +
                    1.0 * emotion_loss
                )
                
                total_loss.backward()
                optimizer.step()
                
                train_losses.append(total_loss.item())
            
            # 검증
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    features = batch['features'].to(device)
                    targets = {key: val.to(device) for key, val in batch['targets'].items()}
                    
                    outputs = self.model(features)
                    
                    material_loss = material_criterion(outputs['materials'], targets['materials'])
                    volatility_loss = volatility_criterion(outputs['volatility'], targets['volatility'].long())
                    duration_loss = duration_criterion(outputs['duration'], targets['duration'])
                    emotion_loss = emotion_criterion(outputs['emotions'], targets['emotions'])
                    
                    total_loss = (
                        2.0 * material_loss +
                        1.0 * volatility_loss +
                        0.5 * duration_loss +
                        1.0 * emotion_loss
                    )
                    
                    val_losses.append(total_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            scheduler.step(avg_val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # 최고 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model_and_preprocessors()
        
        logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    def save_model_and_preprocessors(self):
        """모델과 전처리기 저장"""
        # 모델 저장
        model_path = self.model_save_dir / "movie_scent_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'material_count': self.model.material_count,
                'genre_count': self.model.genre_count,
                'emotion_count': self.model.emotion_count
            }
        }, model_path)
        
        # 전처리기들 저장
        preprocessor_path = self.model_save_dir / "preprocessors.pkl"
        joblib.dump({
            'feature_scaler': self.feature_scaler,
            'material_vocab': self.material_vocab,
            'emotion_vocab': self.emotion_vocab,
            'label_encoders': self.label_encoders
        }, preprocessor_path)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Preprocessors saved to: {preprocessor_path}")
    
    def run_full_training_pipeline(self):
        """전체 훈련 파이프라인 실행"""
        logger.info("Starting full training pipeline...")
        
        # 1. 데이터 로드 및 전처리
        X, y = self.load_and_preprocess_data()
        
        # 2. 데이터 로더 생성
        self.create_data_loaders(X, y)
        
        # 3. 모델 생성
        self.create_model(input_dim=X.shape[1])
        
        # 4. 모델 훈련
        self.train_model(epochs=200, lr=0.001)
        
        logger.info("Training pipeline completed successfully!")

def main():
    """메인 함수"""
    logger.info("🎬 Movie Scent AI - Deep Learning Training Started")
    logger.info("=" * 60)
    
    trainer = MovieScentTrainer()
    trainer.run_full_training_pipeline()
    
    logger.info("✅ Training completed! Model ready for movie scene fragrance prediction.")

if __name__ == "__main__":
    main()