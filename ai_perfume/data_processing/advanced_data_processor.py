from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import json
import ast
import re
from dataclasses import dataclass
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedFragranceData:
    """전처리된 향수 데이터 구조"""
    # 원본 정보
    name: str
    description: str
    rating: float
    rating_count: int
    gender: str
    
    # 처리된 특성들
    main_accords: List[str]
    top_notes: List[str]
    middle_notes: List[str]
    base_notes: List[str]
    
    # 인코딩된 특성들
    accord_encoding: np.ndarray
    note_encoding: np.ndarray
    text_embedding: np.ndarray
    gender_encoding: int
    
    # 품질 지표
    popularity_score: float
    complexity_score: float
    quality_score: float


class AdvancedFragranceDataProcessor:
    """고급 향수 데이터 전처리 시스템"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # 인코더들 초기화
        self.mlb_accords = MultiLabelBinarizer()
        self.mlb_notes = MultiLabelBinarizer()
        self.gender_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # 향료 노트 사전 (NLP 파싱용)
        self.note_patterns = {
            'citrus': ['lemon', 'orange', 'bergamot', 'lime', 'grapefruit', 'mandarin', 'citrus'],
            'floral': ['rose', 'jasmine', 'lavender', 'lily', 'iris', 'peony', 'tuberose', 'violet', 'freesia'],
            'fruity': ['apple', 'strawberry', 'raspberry', 'blackcurrant', 'peach', 'pear', 'plum'],
            'woody': ['cedar', 'sandalwood', 'patchouli', 'vetiver', 'pine', 'cypress', 'guaiac'],
            'spicy': ['pepper', 'cardamom', 'cinnamon', 'nutmeg', 'ginger', 'saffron', 'clove'],
            'herbal': ['mint', 'basil', 'thyme', 'rosemary', 'sage', 'bay'],
            'sweet': ['vanilla', 'caramel', 'honey', 'tonka', 'praline', 'chocolate'],
            'fresh': ['marine', 'aquatic', 'ozone', 'cucumber', 'watermelon'],
            'animalic': ['musk', 'ambergris', 'civet', 'castoreum'],
            'resinous': ['amber', 'benzoin', 'labdanum', 'frankincense', 'myrrh']
        }
        
        # 통계 정보
        self.processing_stats = {}
        
    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        """원시 데이터 로드 및 기본 정리"""
        logger.info("📂 원시 데이터 로드 중...")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"✅ 데이터 로드 완료: {len(df)} 행")
            
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')  # Windows 인코딩
            logger.info(f"✅ 데이터 로드 완료 (CP949): {len(df)} 행")
        
        # 기본 정보 출력
        logger.info(f"📊 컬럼: {list(df.columns)}")
        
        # 데이터 정리
        df_clean = self._clean_basic_data(df)
        
        self.processing_stats['original_count'] = len(df)
        self.processing_stats['cleaned_count'] = len(df_clean)
        self.processing_stats['removed_count'] = len(df) - len(df_clean)
        
        return df_clean
    
    def _clean_basic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 데이터 정리"""
        logger.info("🧹 기본 데이터 정리 중...")
        
        # 컬럼명 정리
        df.columns = df.columns.str.strip()
        
        # 결측치가 많은 행 제거
        df = df.dropna(subset=['Name', 'Description'])
        
        # Rating Value 처리 (문자열로 된 경우 처리)
        df['Rating Value'] = pd.to_numeric(df['Rating Value'], errors='coerce')
        
        # Rating Count 처리 (쉼표 제거)
        df['Rating Count'] = df['Rating Count'].astype(str).str.replace(',', '')
        df['Rating Count'] = pd.to_numeric(df['Rating Count'], errors='coerce')
        
        # 결측치를 기본값으로 대체
        df['Rating Value'] = df['Rating Value'].fillna(3.5)  # 평균 평점
        df['Rating Count'] = df['Rating Count'].fillna(1)    # 최소 카운트
        
        # Main Accords 정리 (문자열을 리스트로 변환)
        df['Main Accords'] = df['Main Accords'].apply(self._parse_list_string)
        
        # Description에서 노트 정보 추출
        df['Top Notes'] = df['Description'].apply(lambda x: self._extract_notes(x, 'top'))
        df['Middle Notes'] = df['Description'].apply(lambda x: self._extract_notes(x, 'middle'))
        df['Base Notes'] = df['Description'].apply(lambda x: self._extract_notes(x, 'base'))
        
        logger.info(f"✅ 정리 완료: {len(df)} 행 유지")
        
        return df
    
    def _parse_list_string(self, list_str: str) -> List[str]:
        """문자열로 된 리스트를 실제 리스트로 변환"""
        if pd.isna(list_str) or not list_str.strip():
            return []
        
        try:
            # ['item1', 'item2'] 형태 파싱
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                return [item.strip().lower() for item in parsed if item.strip()]
        except (ValueError, SyntaxError):
            pass
        
        # 쉼표로 구분된 형태 처리
        if ',' in list_str:
            return [item.strip().lower() for item in list_str.split(',') if item.strip()]
        
        return [list_str.strip().lower()] if list_str.strip() else []
    
    def _extract_notes(self, description: str, note_type: str) -> List[str]:
        """설명에서 노트 정보 추출"""
        if pd.isna(description):
            return []
        
        description = description.lower()
        
        # 노트 타입별 패턴 찾기
        if note_type == 'top':
            patterns = [r'top notes? are? ([^;.]+)', r'opens? with ([^;.]+)']
        elif note_type == 'middle':
            patterns = [r'middle notes? are? ([^;.]+)', r'heart notes? are? ([^;.]+)']
        else:  # base
            patterns = [r'base notes? are? ([^;.]+)', r'dries? down to ([^;.]+)']
        
        extracted_notes = []
        
        for pattern in patterns:
            matches = re.findall(pattern, description)
            for match in matches:
                # 노트들을 파싱 (쉼표와 'and'로 구분)
                notes_text = re.sub(r'\s+and\s+', ', ', match)
                notes = [note.strip() for note in notes_text.split(',')]
                
                # 각 노트를 정리
                for note in notes:
                    note = re.sub(r'\([^)]*\)', '', note).strip()  # 괄호 내용 제거
                    if note and len(note) > 2:
                        extracted_notes.append(note)
        
        return list(set(extracted_notes))  # 중복 제거
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """고급 특성들 생성"""
        logger.info("🔬 고급 특성 생성 중...")
        
        # 1. 인기도 점수 (평점 * log(평점 수))
        df['popularity_score'] = df['Rating Value'] * np.log1p(df['Rating Count'])
        
        # 2. 복합성 점수 (사용된 노트 수)
        df['complexity_score'] = (
            df['Top Notes'].apply(len) + 
            df['Middle Notes'].apply(len) + 
            df['Base Notes'].apply(len)
        )
        
        # 3. 품질 점수 (평점과 복합성의 조합)
        df['quality_score'] = (
            df['Rating Value'] * 0.7 + 
            (df['complexity_score'] / df['complexity_score'].max()) * 3 * 0.3
        )
        
        # 4. 전체 노트 리스트 생성
        df['all_notes'] = df.apply(
            lambda row: row['Top Notes'] + row['Middle Notes'] + row['Base Notes'], 
            axis=1
        )
        
        # 5. 성별 선호도 인코딩
        df['gender_clean'] = df['Gender'].apply(self._clean_gender)
        
        # 6. 브랜드 추출
        df['brand'] = df['Name'].apply(self._extract_brand)
        
        # 7. 텍스트 임베딩 생성
        logger.info("🧠 텍스트 임베딩 생성 중...")
        descriptions = df['Description'].fillna('').tolist()
        embeddings = self.embedding_model.encode(descriptions, show_progress_bar=True)
        df['text_embedding'] = list(embeddings)
        
        # 8. 향료 계열 점수 계산
        for family, keywords in self.note_patterns.items():
            df[f'{family}_score'] = df['all_notes'].apply(
                lambda notes: self._calculate_family_score(notes, keywords)
            )
        
        logger.info("✅ 고급 특성 생성 완료")
        return df
    
    def _clean_gender(self, gender_str: str) -> str:
        """성별 정보 정리"""
        if pd.isna(gender_str):
            return 'unisex'
        
        gender_str = gender_str.lower()
        
        if 'women' in gender_str and 'men' in gender_str:
            return 'unisex'
        elif 'women' in gender_str:
            return 'female'
        elif 'men' in gender_str:
            return 'male'
        else:
            return 'unisex'
    
    def _extract_brand(self, name: str) -> str:
        """제품명에서 브랜드 추출"""
        if pd.isna(name):
            return 'unknown'
        
        # 일반적인 브랜드 패턴들
        brand_patterns = [
            r'([A-Za-z\s]+?)\s+for\s+(women|men)',
            r'by\s+([A-Za-z\s]+?)is',
            r'([A-Za-z\s]+?)\s+Perfumes',
        ]
        
        for pattern in brand_patterns:
            match = re.search(pattern, name)
            if match:
                return match.group(1).strip()
        
        # 마지막 시도: 첫 번째 단어들 사용
        words = name.split()
        if len(words) >= 2:
            return ' '.join(words[-2:]) if 'for' not in words[-2:] else words[0]
        
        return 'unknown'
    
    def _calculate_family_score(self, notes: List[str], keywords: List[str]) -> float:
        """특정 향료 계열의 점수 계산"""
        if not notes:
            return 0.0
        
        matches = 0
        for note in notes:
            for keyword in keywords:
                if keyword in note.lower():
                    matches += 1
                    break
        
        return matches / len(notes)
    
    def encode_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """모든 특성들을 수치형으로 인코딩"""
        logger.info("🔢 특성 인코딩 중...")
        
        # 1. Multi-label 인코딩 (Main Accords)
        accord_matrix = self.mlb_accords.fit_transform(df['Main Accords'])
        logger.info(f"📊 Accords 차원: {accord_matrix.shape[1]}")
        
        # 2. 노트들 인코딩
        all_notes_combined = df['all_notes']
        note_matrix = self.mlb_notes.fit_transform(all_notes_combined)
        logger.info(f"📊 Notes 차원: {note_matrix.shape[1]}")
        
        # 3. 성별 인코딩
        gender_encoded = self.gender_encoder.fit_transform(df['gender_clean'])
        
        # 4. 텍스트 임베딩 스택
        text_embeddings = np.vstack(df['text_embedding'].tolist())
        logger.info(f"📊 Text embedding 차원: {text_embeddings.shape[1]}")
        
        # 5. 수치형 특성들
        numeric_features = df[[
            'Rating Value', 'Rating Count', 'popularity_score', 
            'complexity_score', 'quality_score'
        ] + [f'{family}_score' for family in self.note_patterns.keys()]].values
        
        # 6. 정규화
        numeric_features_scaled = self.scaler.fit_transform(numeric_features)
        
        # 7. 모든 특성 결합
        X = np.hstack([
            accord_matrix,           # Accord features
            note_matrix,            # Note features  
            text_embeddings,        # Text embeddings
            numeric_features_scaled, # Numeric features
            gender_encoded.reshape(-1, 1)  # Gender
        ])
        
        # 8. 타겟 변수 (평점 예측용)
        y = df['Rating Value'].values
        
        # 9. 메타데이터
        metadata = {
            'feature_dimensions': {
                'accords': accord_matrix.shape[1],
                'notes': note_matrix.shape[1], 
                'text_embeddings': text_embeddings.shape[1],
                'numeric': numeric_features_scaled.shape[1],
                'gender': 1,
                'total': X.shape[1]
            },
            'accord_classes': list(self.mlb_accords.classes_),
            'note_classes': list(self.mlb_notes.classes_),
            'gender_classes': list(self.gender_encoder.classes_),
            'processing_stats': self.processing_stats
        }
        
        logger.info(f"✅ 인코딩 완료: {X.shape[0]} 샘플, {X.shape[1]} 특성")
        
        return X, y, metadata
    
    def create_training_data(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> Dict[str, Any]:
        """학습용 데이터 생성"""
        logger.info("📚 학습 데이터 생성 중...")
        
        # 1. Train/Validation/Test 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.18, random_state=42,  # 0.18 * 0.85 ≈ 0.15
            stratify=pd.cut(y_temp, bins=5, labels=False)
        )
        
        # 2. 추가 학습 태스크를 위한 데이터
        
        # 2-1. 성별 예측 태스크
        gender_labels = self.gender_encoder.transform(df['gender_clean'])
        
        # 2-2. 인기도 예측 태스크 (인기도를 범주형으로)
        popularity_labels = pd.cut(
            df['popularity_score'], 
            bins=3, 
            labels=['low', 'medium', 'high']
        ).codes
        
        # 2-3. 복합성 예측 태스크
        complexity_labels = pd.cut(
            df['complexity_score'],
            bins=3,
            labels=['simple', 'medium', 'complex']
        ).codes
        
        # 3. 데이터 텐서 변환
        training_data = {
            'X_train': torch.FloatTensor(X_train),
            'X_val': torch.FloatTensor(X_val),
            'X_test': torch.FloatTensor(X_test),
            
            # 평점 예측
            'y_rating_train': torch.FloatTensor(y_train),
            'y_rating_val': torch.FloatTensor(y_val), 
            'y_rating_test': torch.FloatTensor(y_test),
            
            # 성별 예측  
            'y_gender_train': torch.LongTensor(gender_labels[:len(X_train)]),
            'y_gender_val': torch.LongTensor(gender_labels[len(X_train):len(X_train)+len(X_val)]),
            'y_gender_test': torch.LongTensor(gender_labels[-len(X_test):]),
            
            # 인기도 예측
            'y_popularity_train': torch.LongTensor(popularity_labels[:len(X_train)]),
            'y_popularity_val': torch.LongTensor(popularity_labels[len(X_train):len(X_train)+len(X_val)]),
            'y_popularity_test': torch.LongTensor(popularity_labels[-len(X_test):]),
            
            # 복합성 예측
            'y_complexity_train': torch.LongTensor(complexity_labels[:len(X_train)]),
            'y_complexity_val': torch.LongTensor(complexity_labels[len(X_train):len(X_train)+len(X_val)]),
            'y_complexity_test': torch.LongTensor(complexity_labels[-len(X_test):])
        }
        
        logger.info(f"✅ 학습 데이터 준비 완료:")
        logger.info(f"   📊 Train: {len(X_train)} 샘플")
        logger.info(f"   📊 Validation: {len(X_val)} 샘플")
        logger.info(f"   📊 Test: {len(X_test)} 샘플")
        
        return training_data
    
    def save_processed_data(
        self, 
        training_data: Dict[str, Any], 
        metadata: Dict[str, Any], 
        df: pd.DataFrame,
        output_dir: str = "data/processed"
    ):
        """전처리된 데이터 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 데이터 저장 중: {output_path}")
        
        # 1. 학습 데이터 저장
        torch.save(training_data, output_path / "training_data.pt")
        
        # 2. 메타데이터 저장
        with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 3. 전처리된 DataFrame 저장
        df.to_csv(output_path / "processed_perfume_data.csv", index=False, encoding='utf-8')
        
        # 4. 인코더들 저장
        import joblib
        joblib.dump(self.mlb_accords, output_path / "accord_encoder.pkl")
        joblib.dump(self.mlb_notes, output_path / "note_encoder.pkl")
        joblib.dump(self.gender_encoder, output_path / "gender_encoder.pkl")
        joblib.dump(self.scaler, output_path / "scaler.pkl")
        
        logger.info("✅ 데이터 저장 완료")
        
        # 5. 처리 요약 출력
        self._print_processing_summary(metadata, df)
    
    def _print_processing_summary(self, metadata: Dict[str, Any], df: pd.DataFrame):
        """처리 요약 출력"""
        print("\n" + "="*80)
        print("📊 DATA PROCESSING SUMMARY")
        print("="*80)
        
        print(f"📈 원본 데이터: {metadata['processing_stats']['original_count']:,} 행")
        print(f"✅ 정리된 데이터: {metadata['processing_stats']['cleaned_count']:,} 행")
        print(f"❌ 제거된 데이터: {metadata['processing_stats']['removed_count']:,} 행")
        print(f"📊 특성 차원: {metadata['feature_dimensions']['total']:,}")
        
        print(f"\n🎭 특성 분포:")
        for feature_type, dim in metadata['feature_dimensions'].items():
            if feature_type != 'total':
                print(f"   • {feature_type}: {dim}")
        
        print(f"\n🏷️  클래스 정보:")
        print(f"   • Accords: {len(metadata['accord_classes'])} 종류")
        print(f"   • Notes: {len(metadata['note_classes'])} 종류") 
        print(f"   • Gender: {len(metadata['gender_classes'])} 종류")
        
        print(f"\n📊 데이터 품질 지표:")
        print(f"   • 평균 평점: {df['Rating Value'].mean():.2f}")
        print(f"   • 평균 복합성: {df['complexity_score'].mean():.1f}")
        print(f"   • 평균 인기도: {df['popularity_score'].mean():.1f}")
        
        print("\n🎯 생성된 학습 태스크:")
        print("   • 평점 예측 (회귀)")
        print("   • 성별 분류")
        print("   • 인기도 분류")
        print("   • 복합성 분류")
        
        print("="*80)


def main():
    """메인 데이터 처리 실행"""
    print("🚀 고급 향수 데이터 전처리 시작!")
    
    # 데이터 경로
    raw_data_path = "data/raw/raw_perfume_data.csv"
    
    # 프로세서 초기화
    processor = AdvancedFragranceDataProcessor()
    
    try:
        # 1. 데이터 로드 및 정리
        df_clean = processor.load_and_clean_data(raw_data_path)
        
        # 2. 고급 특성 생성
        df_enhanced = processor.create_advanced_features(df_clean)
        
        # 3. 특성 인코딩
        X, y, metadata = processor.encode_features(df_enhanced)
        
        # 4. 학습 데이터 생성
        training_data = processor.create_training_data(X, y, df_enhanced)
        
        # 5. 데이터 저장
        processor.save_processed_data(training_data, metadata, df_enhanced)
        
        print("\n🎉 데이터 전처리 완료!")
        
    except Exception as e:
        logger.error(f"❌ 데이터 처리 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()