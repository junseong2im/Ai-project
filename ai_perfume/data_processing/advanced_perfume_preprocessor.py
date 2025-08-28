#!/usr/bin/env python3
"""
고급 향수 데이터 전처리 및 딥러닝 훈련 시스템
"""

import pandas as pd
import numpy as np
import json
import re
import ast
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerfumeDataset(Dataset):
    """향수 데이터셋 클래스"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class PerfumeNeuralNetwork(nn.Module):
    """향수 추천을 위한 딥러닝 모델"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        super(PerfumeNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class AdvancedPerfumePreprocessor:
    """고급 향수 데이터 전처리기"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.processed_data = {}
        
        # 전처리에 사용할 도구들
        self.gender_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
        # 향료 노트 매핑
        self.note_categories = {
            'citrus': ['citrus', 'bergamot', 'lemon', 'orange', 'grapefruit', 'lime', 'mandarin'],
            'floral': ['floral', 'rose', 'jasmine', 'lily', 'violet', 'tuberose', 'white floral', 'yellow floral'],
            'woody': ['woody', 'cedar', 'sandalwood', 'pine', 'cypress', 'guaiac wood'],
            'oriental': ['amber', 'vanilla', 'musk', 'oud', 'incense', 'benzoin'],
            'fresh': ['fresh', 'aquatic', 'green', 'herbal', 'mint', 'eucalyptus'],
            'spicy': ['warm spicy', 'fresh spicy', 'cinnamon', 'nutmeg', 'cardamom', 'black pepper'],
            'fruity': ['fruity', 'apple', 'berry', 'peach', 'plum', 'raspberry'],
            'gourmand': ['vanilla', 'chocolate', 'caramel', 'honey', 'coffee', 'almond', 'coconut'],
            'animalic': ['animalic', 'leather', 'musk', 'ambergris']
        }
        
        logger.info("향수 데이터 전처리기 초기화 완료")
    
    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"데이터 로드 완료: {len(self.df)} 개의 향수 데이터")
            return self.df
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """데이터 정리 및 클리닝"""
        logger.info("데이터 클리닝 시작...")
        
        # 기본 클리닝
        self.df = self.df.dropna(subset=['Name', 'Description'])
        
        # 평점 데이터 처리
        self.df['Rating Value'] = pd.to_numeric(self.df['Rating Value'], errors='coerce')
        self.df['Rating Count'] = self.df['Rating Count'].astype(str).str.replace(',', '')
        self.df['Rating Count'] = pd.to_numeric(self.df['Rating Count'], errors='coerce')
        
        # 누락된 평점을 평균으로 대체
        self.df['Rating Value'].fillna(self.df['Rating Value'].mean(), inplace=True)
        self.df['Rating Count'].fillna(0, inplace=True)
        
        # Main Accords 리스트로 변환
        def parse_accords(accords_str):
            try:
                if pd.isna(accords_str) or accords_str == '[]':
                    return []
                return ast.literal_eval(accords_str)
            except:
                return []
        
        self.df['Main Accords'] = self.df['Main Accords'].apply(parse_accords)
        
        logger.info(f"클리닝 완료: {len(self.df)} 개의 향수 데이터 남음")
        return self.df
    
    def extract_features(self) -> Dict[str, np.ndarray]:
        """특성 추출"""
        logger.info("특성 추출 시작...")
        
        features = {}
        
        # 1. 성별 인코딩
        gender_mapping = {
            'for women': 0,
            'for men': 1, 
            'for women and men': 2
        }
        features['gender'] = self.df['Gender'].map(gender_mapping).fillna(2).values
        
        # 2. 평점 특성
        features['rating_value'] = self.df['Rating Value'].values
        features['rating_count_log'] = np.log1p(self.df['Rating Count'].values)
        
        # 3. 텍스트 특성 (설명)
        descriptions = self.df['Description'].fillna('').astype(str)
        description_tfidf = self.tfidf_vectorizer.fit_transform(descriptions).toarray()
        features['description_tfidf'] = description_tfidf
        
        # 4. 향료 노트 카테고리 특성
        note_features = self._extract_note_features()
        features.update(note_features)
        
        # 5. 설명 길이 특성
        features['description_length'] = np.array([len(str(desc)) for desc in descriptions])
        
        # 6. 브랜드 특성 (이름에서 브랜드 추출)
        brands = self.df['Name'].str.split().str[-1]  # 마지막 단어를 브랜드로 가정
        brand_encoder = LabelEncoder()
        features['brand'] = brand_encoder.fit_transform(brands.fillna('Unknown'))
        
        logger.info(f"특성 추출 완료: {len(features)} 개의 특성 그룹")
        return features
    
    def _extract_note_features(self) -> Dict[str, np.ndarray]:
        """향료 노트 기반 특성 추출"""
        note_features = {}
        
        # 각 카테고리별로 특성 생성
        for category, notes in self.note_categories.items():
            category_scores = []
            
            for _, row in self.df.iterrows():
                accords = row['Main Accords']
                if not isinstance(accords, list):
                    accords = []
                
                # 해당 카테고리 노트가 얼마나 포함되어 있는지 계산
                score = sum(1 for accord in accords if any(note in accord.lower() for note in notes))
                category_scores.append(score)
            
            note_features[f'note_{category}'] = np.array(category_scores)
        
        # 총 노트 수
        note_features['total_notes'] = np.array([len(accords) if isinstance(accords, list) else 0 
                                               for accords in self.df['Main Accords']])
        
        return note_features
    
    def create_training_data(self, features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """훈련용 데이터 생성"""
        logger.info("훈련 데이터 생성 중...")
        
        # 모든 특성을 결합
        feature_arrays = []
        feature_names = []
        
        for key, values in features.items():
            if values.ndim == 1:
                feature_arrays.append(values.reshape(-1, 1))
                feature_names.append(key)
            else:
                feature_arrays.append(values)
                feature_names.extend([f"{key}_{i}" for i in range(values.shape[1])])
        
        X = np.concatenate(feature_arrays, axis=1)
        
        # 타겟: 평점 예측 + 카테고리 예측
        y_rating = features['rating_value'].reshape(-1, 1)
        
        # 성별 카테고리를 원-핫 인코딩
        y_gender = np.eye(3)[features['gender'].astype(int)]
        
        # 타겟 결합
        y = np.concatenate([y_rating, y_gender], axis=1)
        
        # 정규화
        X = self.scaler.fit_transform(X)
        
        logger.info(f"훈련 데이터 생성 완료: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def save_preprocessed_data(self, X: np.ndarray, y: np.ndarray, 
                             output_dir: str = "data/processed") -> None:
        """전처리된 데이터 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장
        np.save(output_path / "X_features.npy", X)
        np.save(output_path / "y_targets.npy", y)
        
        # 전처리 도구들 저장
        with open(output_path / "preprocessor_tools.pkl", "wb") as f:
            pickle.dump({
                'scaler': self.scaler,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'note_categories': self.note_categories
            }, f)
        
        # 메타데이터 저장
        metadata = {
            'feature_dim': X.shape[1],
            'target_dim': y.shape[1],
            'num_samples': X.shape[0],
            'preprocessing_info': {
                'tfidf_features': self.tfidf_vectorizer.get_feature_names_out().tolist() if hasattr(self.tfidf_vectorizer, 'get_feature_names_out') else [],
                'note_categories': list(self.note_categories.keys())
            }
        }
        
        with open(output_path / "metadata.json", "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"전처리된 데이터 저장 완료: {output_path}")
    
    def train_deep_learning_model(self, X: np.ndarray, y: np.ndarray, 
                                model_save_path: str = "models/perfume_dl_model.pth") -> Dict[str, Any]:
        """딥러닝 모델 훈련"""
        logger.info("딥러닝 모델 훈련 시작...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 데이터셋 생성
        train_dataset = PerfumeDataset(X_train, y_train)
        test_dataset = PerfumeDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 모델 생성
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        hidden_dims = [256, 128, 64]
        
        model = PerfumeNeuralNetwork(input_dim, hidden_dims, output_dim)
        
        # 손실 함수와 옵티마이저
        criterion_rating = nn.MSELoss()
        criterion_category = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # 훈련
        num_epochs = 100
        best_loss = float('inf')
        training_history = {'train_loss': [], 'test_loss': []}
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = model(batch_x)
                
                # 평점 예측 손실
                rating_loss = criterion_rating(outputs[:, 0:1], batch_y[:, 0:1])
                
                # 카테고리 예측 손실 (성별)
                category_loss = criterion_category(outputs[:, 1:4], 
                                                 torch.argmax(batch_y[:, 1:4], dim=1))
                
                # 총 손실
                total_loss = rating_loss + 0.5 * category_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # 검증
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x)
                    rating_loss = criterion_rating(outputs[:, 0:1], batch_y[:, 0:1])
                    category_loss = criterion_category(outputs[:, 1:4], 
                                                     torch.argmax(batch_y[:, 1:4], dim=1))
                    total_loss = rating_loss + 0.5 * category_loss
                    test_loss += total_loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            
            training_history['train_loss'].append(avg_train_loss)
            training_history['test_loss'].append(avg_test_loss)
            
            scheduler.step(avg_test_loss)
            
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                    'model_config': {
                        'input_dim': input_dim,
                        'hidden_dims': hidden_dims,
                        'output_dim': output_dim
                    }
                }, model_save_path)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                          f'Train Loss: {avg_train_loss:.4f}, '
                          f'Test Loss: {avg_test_loss:.4f}')
        
        logger.info(f"모델 훈련 완료. 최고 성능: {best_loss:.4f}")
        
        return {
            'best_loss': best_loss,
            'training_history': training_history,
            'model_path': model_save_path,
            'test_performance': self._evaluate_model(model, test_loader)
        }
    
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """모델 평가"""
        model.eval()
        rating_errors = []
        category_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                
                # 평점 예측 오차
                rating_pred = outputs[:, 0:1]
                rating_true = batch_y[:, 0:1]
                rating_errors.extend(torch.abs(rating_pred - rating_true).cpu().numpy())
                
                # 카테고리 정확도
                category_pred = torch.argmax(outputs[:, 1:4], dim=1)
                category_true = torch.argmax(batch_y[:, 1:4], dim=1)
                category_correct += (category_pred == category_true).sum().item()
                total_samples += batch_y.shape[0]
        
        return {
            'rating_mae': np.mean(rating_errors),
            'rating_rmse': np.sqrt(np.mean(np.square(rating_errors))),
            'category_accuracy': category_correct / total_samples
        }

def main():
    """메인 실행 함수"""
    try:
        # 데이터 경로 설정
        data_path = "C:/Users/user/Desktop/ai project/ai_perfume/data/raw/raw_perfume_data.csv"
        
        # 전처리기 초기화
        preprocessor = AdvancedPerfumePreprocessor(data_path)
        
        # 데이터 로드 및 클리닝
        preprocessor.load_data()
        preprocessor.clean_data()
        
        # 특성 추출
        features = preprocessor.extract_features()
        
        # 훈련 데이터 생성
        X, y = preprocessor.create_training_data(features)
        
        # 전처리된 데이터 저장
        preprocessor.save_preprocessed_data(X, y)
        
        # 딥러닝 모델 훈련
        model_save_path = "C:/Users/user/Desktop/ai project/ai_perfume/models/perfume_dl_model.pth"
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        
        training_results = preprocessor.train_deep_learning_model(X, y, model_save_path)
        
        # 결과 출력
        logger.info("="*80)
        logger.info("🎉 딥러닝 모델 훈련 완료!")
        logger.info("="*80)
        logger.info(f"📊 최종 손실값: {training_results['best_loss']:.4f}")
        logger.info(f"📈 평점 예측 MAE: {training_results['test_performance']['rating_mae']:.4f}")
        logger.info(f"📈 평점 예측 RMSE: {training_results['test_performance']['rating_rmse']:.4f}")
        logger.info(f"🎯 카테고리 정확도: {training_results['test_performance']['category_accuracy']:.3f}")
        logger.info(f"💾 모델 저장 위치: {training_results['model_path']}")
        
        # 훈련 기록 저장
        with open("C:/Users/user/Desktop/ai project/ai_perfume/models/training_results.json", "w") as f:
            json.dump({
                'best_loss': training_results['best_loss'],
                'test_performance': training_results['test_performance'],
                'model_path': training_results['model_path']
            }, f, indent=2)
        
        logger.info("✅ 모든 작업이 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()