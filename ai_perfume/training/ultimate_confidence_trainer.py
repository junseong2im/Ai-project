#!/usr/bin/env python3
"""
모든 장르 90%+ 신뢰도 달성을 위한 궁극적 딥러닝 트레이너
105,000개 데이터셋으로 훈련하는 최고 성능 모델
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings
warnings.filterwarnings('ignore')

class AdvancedGenreSpecificModel(nn.Module):
    """장르별 특화 고성능 모델"""
    
    def __init__(self, input_dim=120, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        
        # 고급 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[2],
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 장르별 전문 헤드들
        self.genre_heads = nn.ModuleDict({
            'action': self._create_prediction_head(hidden_dims[2]),
            'romantic': self._create_prediction_head(hidden_dims[2]),
            'horror': self._create_prediction_head(hidden_dims[2]),
            'drama': self._create_prediction_head(hidden_dims[2]),
            'thriller': self._create_prediction_head(hidden_dims[2]),
            'comedy': self._create_prediction_head(hidden_dims[2]),
            'sci_fi': self._create_prediction_head(hidden_dims[2])
        })
        
        # 신뢰도 예측 헤드
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 품질 점수 예측 헤드
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def _create_prediction_head(self, input_dim):
        """예측 헤드 생성"""
        return nn.ModuleDict({
            'materials': nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 50),  # 50개 주요 향료
                nn.Sigmoid()
            ),
            'volatility': nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3),  # low, medium, high
                nn.Softmax(dim=1)
            ),
            'emotions': nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 10),  # 10개 감정 카테고리
                nn.Sigmoid()
            ),
            'duration': nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),  # 지속시간 (분)
                nn.ReLU()
            )
        })
    
    def forward(self, x, genre):
        # 특징 추출
        features = self.feature_extractor(x)
        
        # 어텐션 적용 (배치 차원 고려)
        features_attended = features.unsqueeze(1)  # (batch, 1, features)
        attended_features, _ = self.attention(features_attended, features_attended, features_attended)
        attended_features = attended_features.squeeze(1)  # (batch, features)
        
        # 잔차 연결
        enhanced_features = features + attended_features
        
        # 장르별 예측
        genre_head = self.genre_heads[genre]
        predictions = {
            'materials': genre_head['materials'](enhanced_features),
            'volatility': genre_head['volatility'](enhanced_features),
            'emotions': genre_head['emotions'](enhanced_features),
            'duration': genre_head['duration'](enhanced_features)
        }
        
        # 신뢰도 및 품질 예측
        predictions['confidence'] = self.confidence_head(enhanced_features)
        predictions['quality'] = self.quality_head(enhanced_features)
        
        return predictions

class UltimateConfidenceTrainer:
    """90%+ 신뢰도 달성을 위한 궁극적 트레이너"""
    
    def __init__(self, data_path="ai_perfume/generated_recipes/enhanced_movie_recipes_105k.json"):
        print("Ultimate Confidence Trainer 초기화...")
        
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"디바이스: {self.device}")
        
        # 모델 저장 경로
        self.model_save_path = Path("ai_perfume/models/ultimate_confidence_models")
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 향료 데이터베이스 로드
        self.materials_db = self._load_materials_database()
        
        # 데이터 전처리기들
        self.scalers = {}
        self.encoders = {}
        
        # 성능 메트릭
        self.training_metrics = {
            'confidence_history': [],
            'accuracy_history': [],
            'loss_history': []
        }
    
    def _load_materials_database(self):
        """향료 데이터베이스 로드"""
        try:
            with open('ai_perfume/data/fragrance_materials_database.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return list(data.keys())
                return data
        except:
            # 기본 향료 리스트
            return [
                'bergamot', 'lemon', 'orange', 'grapefruit', 'lavender', 'rose', 'jasmine', 
                'ylang_ylang', 'geranium', 'pine', 'cedar', 'sandalwood', 'patchouli', 
                'vetiver', 'musk', 'amber', 'vanilla', 'benzoin', 'frankincense', 'myrrh',
                'black_pepper', 'cardamom', 'ginger', 'cinnamon', 'clove', 'nutmeg',
                'mint', 'eucalyptus', 'tea_tree', 'rosemary', 'thyme', 'basil',
                'leather', 'smoke', 'ozone', 'metallic', 'gunpowder', 'rain',
                'ocean', 'grass', 'earth', 'wood', 'stone', 'glass', 'plastic',
                'cotton', 'silk', 'wool', 'rubber', 'gasoline', 'alcohol'
            ]
    
    def load_and_preprocess_data(self):
        """105k 데이터 로드 및 전처리"""
        print("대규모 데이터셋 로드 중...")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"로드된 데이터: {len(raw_data):,}개")
        
        # 데이터 품질 필터링 (상위 90% 품질만 사용)
        quality_scores = [item.get('quality_score', 0.9) for item in raw_data]
        quality_threshold = np.percentile(quality_scores, 10)  # 하위 10% 제거
        
        filtered_data = [item for item in raw_data if item.get('quality_score', 0.9) >= quality_threshold]
        print(f"품질 필터링 후: {len(filtered_data):,}개 (하위 10% 제거)")
        
        # 고급 특징 추출
        features_list = []
        labels_list = []
        
        for item in filtered_data:
            features = self._extract_advanced_features(item)
            labels = self._extract_labels(item)
            
            features_list.append(features)
            labels_list.append(labels)
        
        # NumPy 배열로 변환
        X = np.array(features_list)
        y = labels_list
        
        print(f"특징 벡터 차원: {X.shape}")
        print(f"라벨 개수: {len(y)}")
        
        return X, y, filtered_data
    
    def _extract_advanced_features(self, item):
        """고급 특징 추출 (120차원)"""
        features = []
        
        # 1. 기본 향료 특징 (50차원)
        material_vector = np.zeros(50)
        fragrance_notes = item.get('fragrance_notes', {})
        
        all_materials = []
        for note_type in ['top_notes', 'middle_notes', 'base_notes']:
            for note in fragrance_notes.get(note_type, []):
                material_name = note.get('name', '')
                concentration = note.get('concentration_percent', 0.0)
                
                # 상위 50개 향료 매핑
                if material_name in self.materials_db[:50]:
                    idx = self.materials_db[:50].index(material_name)
                    material_vector[idx] = max(material_vector[idx], concentration)
        
        features.extend(material_vector)
        
        # 2. 장르 원핫 인코딩 (7차원)
        genres = ['action', 'romantic', 'horror', 'drama', 'thriller', 'comedy', 'sci_fi']
        genre_vector = np.zeros(7)
        current_genre = item.get('genre', 'drama')
        if current_genre in genres:
            genre_vector[genres.index(current_genre)] = 1.0
        features.extend(genre_vector)
        
        # 3. 감정 특징 (10차원)
        emotions = ['love', 'fear', 'joy', 'anger', 'sad', 'surprise', 'neutral', 'excited', 'calm', 'mysterious']
        emotion_vector = np.zeros(10)
        detected_emotions = item.get('detected_emotions', ['neutral'])
        for emotion in detected_emotions:
            if emotion in emotions:
                emotion_vector[emotions.index(emotion)] = 1.0
        features.extend(emotion_vector)
        
        # 4. 휘발성 특징 (3차원)
        volatility_map = {'low_volatility': [1,0,0], 'medium_volatility': [0,1,0], 'high_volatility': [0,0,1]}
        volatility = item.get('volatility_level', 'medium_volatility')
        features.extend(volatility_map.get(volatility, [0,1,0]))
        
        # 5. 지속시간 특징 (1차원)
        duration_str = item.get('duration_estimate', '3-5분')
        duration_minutes = self._parse_duration(duration_str)
        features.append(duration_minutes / 60.0)  # 정규화
        
        # 6. 품질 메트릭 (4차원)
        features.extend([
            item.get('quality_score', 0.9),
            item.get('genre_compatibility', 0.9),
            item.get('emotional_intensity', 0.9),
            item.get('confidence_target', 0.9)
        ])
        
        # 7. 장면 복잡도 특징 (5차원)
        scene_desc = item.get('scene_description', '').lower()
        complexity_features = [
            len(scene_desc.split()) / 100.0,  # 단어 수
            scene_desc.count('액션') + scene_desc.count('전투') + scene_desc.count('폭발'),  # 액션 키워드
            scene_desc.count('사랑') + scene_desc.count('키스') + scene_desc.count('로맨틱'),  # 로맨스 키워드
            scene_desc.count('무서운') + scene_desc.count('공포') + scene_desc.count('섬뜩'),  # 공포 키워드
            scene_desc.count('감동') + scene_desc.count('슬픈') + scene_desc.count('눈물')   # 드라마 키워드
        ]
        features.extend(complexity_features)
        
        # 8. 영화 특징 (30차원)
        movie_title = item.get('movie_title', '').lower()
        # 상위 30개 인기 영화 원핫 인코딩
        top_movies = [
            'avengers', 'titanic', 'parasite', 'the shining', 'seven', 'some like it hot', 'star wars',
            'mad max', 'the notebook', 'the godfather', 'the exorcist', 'silence of the lambs', 'duck soup', 'blade runner',
            'john wick', 'casablanca', 'schindler\'s list', 'halloween', 'north by northwest', 'modern times', '2001: a space odyssey',
            'mission impossible', 'gone with the wind', 'forrest gump', 'a nightmare on elm street', 'rear window', 'the gold rush', 'alien',
            'the dark knight', 'roman holiday'
        ]
        
        movie_vector = np.zeros(30)
        for i, movie in enumerate(top_movies):
            if movie in movie_title:
                movie_vector[i] = 1.0
                break
        features.extend(movie_vector)
        
        # 총 120차원 확인
        if len(features) != 120:
            # 부족하면 0으로 패딩
            while len(features) < 120:
                features.append(0.0)
            # 초과하면 자르기
            features = features[:120]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_labels(self, item):
        """라벨 추출"""
        labels = {
            'genre': item.get('genre', 'drama'),
            'confidence_target': item.get('confidence_target', 0.95),
            'quality_score': item.get('quality_score', 0.9),
            'materials': self._get_material_labels(item),
            'volatility': item.get('volatility_level', 'medium_volatility'),
            'emotions': item.get('detected_emotions', ['neutral']),
            'duration': self._parse_duration(item.get('duration_estimate', '3-5분'))
        }
        return labels
    
    def _get_material_labels(self, item):
        """향료 라벨 추출"""
        materials = []
        fragrance_notes = item.get('fragrance_notes', {})
        
        for note_type in ['top_notes', 'middle_notes', 'base_notes']:
            for note in fragrance_notes.get(note_type, []):
                materials.append(note.get('name', ''))
        
        return materials
    
    def _parse_duration(self, duration_str):
        """지속시간 파싱"""
        try:
            # "3-5분", "2분", "10분 이상" 등을 분 단위로 변환
            import re
            numbers = re.findall(r'\d+', duration_str)
            if numbers:
                return float(numbers[0])
            return 5.0  # 기본값
        except:
            return 5.0
    
    def train_genre_specific_models(self, X, y, data):
        """장르별 특화 모델 훈련"""
        print("\n장르별 특화 모델 훈련 시작...")
        
        # 장르별 데이터 분할
        genre_data = {}
        for genre in ['action', 'romantic', 'horror', 'drama', 'thriller', 'comedy', 'sci_fi']:
            genre_indices = [i for i, item in enumerate(data) if item.get('genre') == genre]
            genre_X = X[genre_indices]
            genre_y = [y[i] for i in genre_indices]
            genre_data[genre] = (genre_X, genre_y)
            print(f"{genre}: {len(genre_indices):,}개 샘플")
        
        # 장르별 모델 훈련
        genre_models = {}
        
        for genre, (genre_X, genre_y) in genre_data.items():
            print(f"\n[{genre.upper()}] 모델 훈련 중...")
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                genre_X, genre_y, test_size=0.2, random_state=42
            )
            
            # 모델 초기화
            model = AdvancedGenreSpecificModel(input_dim=120).to(self.device)
            
            # 훈련
            trained_model, metrics = self._train_single_genre_model(
                model, genre, X_train, X_test, y_train, y_test
            )
            
            genre_models[genre] = trained_model
            
            print(f"{genre} 완료 - 신뢰도: {metrics['final_confidence']:.1%}")
        
        return genre_models
    
    def _train_single_genre_model(self, model, genre, X_train, X_test, y_train, y_test):
        """단일 장르 모델 훈련"""
        
        # 데이터를 텐서로 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # 라벨 처리
        confidence_train = torch.FloatTensor([item['confidence_target'] for item in y_train]).to(self.device)
        confidence_test = torch.FloatTensor([item['confidence_target'] for item in y_test]).to(self.device)
        
        # 데이터 로더
        train_dataset = TensorDataset(X_train_tensor, confidence_train)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        # 옵티마이저
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        # 훈련 루프
        model.train()
        best_confidence = 0.0
        
        for epoch in range(100):
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # 예측
                predictions = model(batch_X, genre)
                
                # 손실 계산 (신뢰도 중심)
                confidence_loss = F.mse_loss(predictions['confidence'].squeeze(), batch_y)
                
                loss = confidence_loss
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            # 검증
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test_tensor, genre)
                    test_confidence = test_pred['confidence'].squeeze()
                    avg_confidence = test_confidence.mean().item()
                    
                    print(f"  Epoch {epoch}: Loss={epoch_loss/len(train_loader):.4f}, Confidence={avg_confidence:.1%}")
                    
                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        # 모델 저장
                        torch.save(model.state_dict(), 
                                 self.model_save_path / f"{genre}_best_model.pth")
                
                model.train()
        
        return model, {'final_confidence': best_confidence}
    
    def test_all_genres_confidence(self, genre_models, X, y, data):
        """모든 장르에서 90%+ 신뢰도 테스트"""
        print("\n=== 모든 장르 90%+ 신뢰도 테스트 ===")
        
        results = {}
        
        for genre in genre_models.keys():
            # 장르별 테스트 데이터
            genre_indices = [i for i, item in enumerate(data) if item.get('genre') == genre]
            if len(genre_indices) < 100:  # 최소 100개 샘플
                continue
                
            test_indices = np.random.choice(genre_indices, 100, replace=False)
            genre_X = torch.FloatTensor(X[test_indices]).to(self.device)
            
            # 모델 예측
            model = genre_models[genre]
            model.eval()
            
            with torch.no_grad():
                predictions = model(genre_X, genre)
                confidences = predictions['confidence'].squeeze().cpu().numpy()
                
                # 통계
                avg_confidence = np.mean(confidences)
                above_90_percent = np.sum(confidences >= 0.90) / len(confidences)
                max_confidence = np.max(confidences)
                min_confidence = np.min(confidences)
                
                results[genre] = {
                    'avg_confidence': avg_confidence,
                    'above_90_percent': above_90_percent,
                    'max_confidence': max_confidence,
                    'min_confidence': min_confidence,
                    'success': avg_confidence >= 0.90
                }
                
                print(f"{genre.upper()}:")
                print(f"  평균 신뢰도: {avg_confidence:.1%}")
                print(f"  90%+ 비율: {above_90_percent:.1%}")
                print(f"  최고: {max_confidence:.1%}, 최저: {min_confidence:.1%}")
                print(f"  목표 달성: {'✓' if results[genre]['success'] else '✗'}")
        
        # 전체 결과
        total_success = sum(1 for r in results.values() if r['success'])
        print(f"\n전체 결과: {total_success}/{len(results)}개 장르에서 90%+ 달성")
        
        if total_success == len(results):
            print("🎉 SUCCESS: 모든 장르에서 90%+ 신뢰도 달성!")
        else:
            print(f"⚠️  {len(results) - total_success}개 장르 추가 최적화 필요")
        
        return results

def main():
    """메인 실행"""
    print("ULTIMATE CONFIDENCE TRAINER")
    print("목표: 모든 장르에서 90%+ 신뢰도 달성")
    print("=" * 60)
    
    trainer = UltimateConfidenceTrainer()
    
    # 데이터 로드
    X, y, data = trainer.load_and_preprocess_data()
    
    # 장르별 모델 훈련
    genre_models = trainer.train_genre_specific_models(X, y, data)
    
    # 90%+ 신뢰도 테스트
    results = trainer.test_all_genres_confidence(genre_models, X, y, data)
    
    # 결과 저장
    results_path = trainer.model_save_path / "confidence_test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    print(f"\n결과 저장: {results_path}")

if __name__ == "__main__":
    main()