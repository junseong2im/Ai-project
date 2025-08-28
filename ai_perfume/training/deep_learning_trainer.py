#!/usr/bin/env python3
"""
향료 원료 딥러닝 훈련 시스템
200,000개 데이터셋으로 편향 없는 AI 모델 구축
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FragranceDataset(Dataset):
    """향료 원료 데이터셋 클래스"""
    
    def __init__(self, data_path: str, split: str = "train", shared_encoders: Dict = None, shared_scaler = None):
        self.split = split
        self.data_path = Path(data_path)
        
        # 데이터 로드
        if split == "train":
            file_path = self.data_path / "fragrance_train.json"
        elif split == "validation":
            file_path = self.data_path / "fragrance_validation.json"
        else:
            file_path = self.data_path / "fragrance_test.json"
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.materials = data["materials"]
        
        # 라벨 인코더 초기화 (공유 또는 새로 생성)
        if shared_encoders:
            self.family_encoder = shared_encoders['family']
            self.volatility_encoder = shared_encoders['volatility']
            self.extraction_encoder = shared_encoders['extraction']
            self.origin_encoder = shared_encoders['origin']
        else:
            self.family_encoder = LabelEncoder()
            self.volatility_encoder = LabelEncoder()
            self.extraction_encoder = LabelEncoder()
            self.origin_encoder = LabelEncoder()
        
        # 특성 추출 및 전처리
        self.features, self.labels = self._preprocess_data()
        
        # 스케일러 (공유 또는 새로 생성)
        if shared_scaler:
            self.scaler = shared_scaler
            self.features = self.scaler.transform(self.features)
        else:
            self.scaler = StandardScaler()
            if split == "train":
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.features = self.scaler.transform(self.features)
            
        # 텐서 변환
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(self.labels)
        
        print(f"Loaded {split} dataset: {len(self.materials)} samples")
        print(f"Feature shape: {self.features.shape}")
        print(f"Label shape: {self.labels.shape}")
    
    def _preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 전처리 및 특성 추출"""
        features = []
        labels = []
        
        # 라벨 인코더 훈련 (훈련 데이터셋에서만 실행)
        if self.split == "train":
            # 모든 카테고리 값 수집 (라벨 인코더 훈련용)
            families = [mat["family"] for mat in self.materials]
            volatilities = [mat["olfactory_properties"]["volatility"] for mat in self.materials]
            extractions = [mat["production_info"]["extraction_method"] for mat in self.materials]
            origins = [mat["origin_type"] for mat in self.materials]
            
            # 라벨 인코더 훈련
            self.family_encoder.fit(families)
            self.volatility_encoder.fit(volatilities)
            self.extraction_encoder.fit(extractions)
            self.origin_encoder.fit(origins)
        
        for material in self.materials:
            # 입력 특성 추출
            feature_vector = []
            
            # 1. 화학적 특성
            feature_vector.append(material.get("molecular_weight", 150))
            feature_vector.append(material["physical_properties"]["boiling_point_c"])
            feature_vector.append(material["physical_properties"]["density_g_ml"])
            feature_vector.append(material["physical_properties"]["refractive_index"])
            feature_vector.append(material["physical_properties"]["flash_point_c"])
            feature_vector.append(material["physical_properties"]["solubility_ethanol"])
            
            # 2. 범주형 변수 인코딩
            feature_vector.append(self.family_encoder.transform([material["family"]])[0])
            feature_vector.append(self.volatility_encoder.transform([material["olfactory_properties"]["volatility"]])[0])
            feature_vector.append(self.extraction_encoder.transform([material["production_info"]["extraction_method"]])[0])
            feature_vector.append(self.origin_encoder.transform([material["origin_type"]])[0])
            
            # 3. 이진 특성
            feature_vector.append(1.0 if material["is_natural"] else 0.0)
            feature_vector.append(1.0 if material["regulatory_info"]["ifra_restricted"] else 0.0)
            feature_vector.append(1.0 if material["regulatory_info"]["allergen_declaration"] else 0.0)
            
            # 4. 경제적 특성
            feature_vector.append(material["economic_data"]["price_usd_per_kg"])
            feature_vector.append(material["economic_data"]["availability_score"])
            feature_vector.append(material["economic_data"]["annual_production_tons"])
            
            # 5. 생산 특성
            feature_vector.append(material["production_info"]["yield_percentage"])
            feature_vector.append(material["production_info"]["purity_percentage"])
            feature_vector.append(material["production_info"]["processing_time_hours"])
            
            # 6. 응용 분야 (이진)
            feature_vector.append(1.0 if material["applications"]["fine_fragrance"] else 0.0)
            feature_vector.append(1.0 if material["applications"]["personal_care"] else 0.0)
            feature_vector.append(1.0 if material["applications"]["home_care"] else 0.0)
            feature_vector.append(1.0 if material["applications"]["air_care"] else 0.0)
            
            # 7. 향기 설명자 수 (복잡도 지표)
            feature_vector.append(len(material["olfactory_properties"]["odor_descriptors"]))
            
            features.append(feature_vector)
            
            # 목표 레이블 (예측할 향기 특성)
            label_vector = [
                material["olfactory_properties"]["intensity"],
                material["olfactory_properties"]["longevity_hours"],
                material["olfactory_properties"]["diffusion"],
                material["olfactory_properties"]["threshold_ppb"],
                material["regulatory_info"]["max_concentration_percent"]
            ]
            
            labels.append(label_vector)
        
        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FragranceNet(nn.Module):
    """향료 원료 예측 신경망 모델"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super(FragranceNet, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        layers = []
        current_dim = input_dim
        
        # 히든 레이어 구성
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        # 출력 레이어
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class FragranceTrainer:
    """향료 원료 딥러닝 훈련 클래스"""
    
    def __init__(self, data_dir: str, model_save_dir: str):
        self.data_dir = Path(data_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 데이터셋 로드 (훈련 데이터를 기준으로 인코더와 스케일러 공유)
        self.train_dataset = FragranceDataset(data_dir, "train")
        
        # 훈련 데이터의 인코더와 스케일러를 검증/테스트 데이터에서 공유
        shared_encoders = {
            'family': self.train_dataset.family_encoder,
            'volatility': self.train_dataset.volatility_encoder,
            'extraction': self.train_dataset.extraction_encoder,
            'origin': self.train_dataset.origin_encoder
        }
        
        self.val_dataset = FragranceDataset(
            data_dir, "validation", 
            shared_encoders=shared_encoders,
            shared_scaler=self.train_dataset.scaler
        ) 
        self.test_dataset = FragranceDataset(
            data_dir, "test",
            shared_encoders=shared_encoders, 
            shared_scaler=self.train_dataset.scaler
        )
        
        # 데이터로더 설정
        self.batch_size = 128
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        # 모델 초기화
        input_dim = self.train_dataset.features.shape[1]
        output_dim = self.train_dataset.labels.shape[1]
        
        self.model = FragranceNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[1024, 512, 256, 128, 64]
        ).to(device)
        
        # 손실 함수 및 옵티마이저
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=10,
            factor=0.5
        )
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self) -> float:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features, labels = features.to(device), labels.to(device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.6f}'
                )
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, epochs: int = 100, early_stopping_patience: int = 20):
        """모델 훈련"""
        self.logger.info(f"Starting training for {epochs} epochs...")
        
        patience_counter = 0
        start_time = datetime.now()
        
        for epoch in range(epochs):
            epoch_start = datetime.now()
            
            # 훈련
            train_loss = self.train_epoch()
            
            # 검증
            val_loss = self.validate()
            
            # 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 스케줄러 업데이트
            self.scheduler.step(val_loss)
            
            # 최고 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model("best_model.pth", epoch, train_loss, val_loss)
                patience_counter = 0
            else:
                patience_counter += 1
            
            epoch_time = datetime.now() - epoch_start
            
            self.logger.info(
                f'Epoch {epoch+1}/{epochs} - '
                f'Train Loss: {train_loss:.6f}, '
                f'Val Loss: {val_loss:.6f}, '
                f'Time: {epoch_time.total_seconds():.1f}s'
            )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # 중간 모델 저장 (매 10 에포크)
            if (epoch + 1) % 10 == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pth", epoch, train_loss, val_loss)
        
        total_time = datetime.now() - start_time
        self.logger.info(f"Training completed in {total_time}")
        
        # 훈련 곡선 시각화
        self.plot_training_curves()
        
        # 최종 테스트
        self.test_model()
    
    def save_model(self, filename: str, epoch: int, train_loss: float, val_loss: float):
        """모델 저장"""
        save_path = self.model_save_dir / filename
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'scaler': self.train_dataset.scaler,
            'label_encoders': {
                'family': self.train_dataset.family_encoder,
                'volatility': self.train_dataset.volatility_encoder,
                'extraction': self.train_dataset.extraction_encoder,
                'origin': self.train_dataset.origin_encoder
            }
        }, save_path)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """모델 로드"""
        load_path = self.model_save_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Model loaded from {load_path}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
    
    def test_model(self):
        """모델 테스트"""
        self.logger.info("Testing model on test set...")
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in self.test_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_test_loss = total_loss / len(self.test_loader)
        
        # 예측 정확도 계산
        predictions = np.vstack(all_predictions)
        true_labels = np.vstack(all_labels)
        
        # 각 출력에 대한 MAE 계산
        maes = np.mean(np.abs(predictions - true_labels), axis=0)
        
        self.logger.info(f"Test Loss (MSE): {avg_test_loss:.6f}")
        self.logger.info("Mean Absolute Errors by output:")
        output_names = ["Intensity", "Longevity", "Diffusion", "Threshold", "Max Concentration"]
        for i, (name, mae) in enumerate(zip(output_names, maes)):
            self.logger.info(f"  {name}: {mae:.4f}")
    
    def plot_training_curves(self):
        """훈련 곡선 시각화"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.yscale('log')
        plt.title('Training and Validation Loss (Log Scale)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / "training_curves.png", dpi=150)
        plt.show()
        
        self.logger.info(f"Training curves saved to {self.model_save_dir / 'training_curves.png'}")

def main():
    """메인 훈련 함수"""
    # 데이터 경로 설정
    data_dir = "../data/datasets"
    model_save_dir = "../models/fragrance_dl_models"
    
    # 훈련기 초기화
    trainer = FragranceTrainer(data_dir, model_save_dir)
    
    # 모델 훈련
    trainer.train(epochs=50, early_stopping_patience=15)
    
    print("\nTraining completed!")
    print(f"Best model saved in: {trainer.model_save_dir}")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")

if __name__ == "__main__":
    main()