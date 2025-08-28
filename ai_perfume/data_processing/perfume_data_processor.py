from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class ProcessingConfig:
    """데이터 처리 설정"""
    min_text_length: int = 10
    max_text_length: int = 1000
    min_rating: float = 1.0
    max_rating: float = 10.0
    vectorizer_max_features: int = 5000
    

class PerfumeDataProcessor:
    """향수 데이터 전처리 및 변환 클래스"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None) -> None:
        self.config = config or ProcessingConfig()
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.text_vectorizer = TfidfVectorizer(
            max_features=self.config.vectorizer_max_features,
            stop_words=None,  # 한국어는 별도 불용어 처리 필요
            ngram_range=(1, 2)
        )
        
    def load_raw_data(self, data_path: str | Path) -> pd.DataFrame:
        """원시 데이터 로드"""
        data_path = Path(data_path)
        
        if data_path.suffix == '.json':
            with data_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif data_path.suffix == '.csv':
            return pd.read_csv(data_path, encoding='utf-8-sig')
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {data_path.suffix}")
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not isinstance(text, str):
            return ""
        
        # 기본적인 텍스트 정리
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text)  # 특수문자 제거 (한글 유지)
        
        return text
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 유효성 검증 및 필터링"""
        initial_count = len(df)
        
        # 텍스트 길이 필터링
        if 'text' in df.columns:
            df['text_length'] = df['text'].astype(str).str.len()
            df = df[
                (df['text_length'] >= self.config.min_text_length) &
                (df['text_length'] <= self.config.max_text_length)
            ]
        
        # 평점 범위 필터링
        if 'rating' in df.columns:
            df = df[
                (df['rating'] >= self.config.min_rating) &
                (df['rating'] <= self.config.max_rating)
            ]
        
        # 중복 제거
        if 'text' in df.columns:
            df = df.drop_duplicates(subset=['text'])
        
        final_count = len(df)
        print(f"데이터 유효성 검증 완료: {initial_count} -> {final_count} ({final_count/initial_count*100:.1f}%)")
        
        return df.reset_index(drop=True)
    
    def process_notes(self, notes_data: List[str] | str) -> List[str]:
        """노트 데이터 처리"""
        if isinstance(notes_data, str):
            # JSON 문자열인 경우 파싱
            try:
                notes_data = json.loads(notes_data)
            except json.JSONDecodeError:
                # 콤마로 구분된 문자열인 경우
                notes_data = [note.strip() for note in notes_data.split(',')]
        
        if not isinstance(notes_data, list):
            return []
        
        # 노트 정리
        processed_notes = []
        for note in notes_data:
            if isinstance(note, str) and note.strip():
                processed_notes.append(note.strip().lower())
        
        return processed_notes
    
    def extract_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """특성 추출"""
        features = {}
        
        # 텍스트 특성
        if 'text' in df.columns:
            # 텍스트 정리
            df['cleaned_text'] = df['text'].apply(self.clean_text)
            
            # TF-IDF 벡터화
            text_features = self.text_vectorizer.fit_transform(df['cleaned_text'])
            features['text_tfidf'] = text_features.toarray()
            
            # 텍스트 통계
            features['text_stats'] = {
                'length': df['cleaned_text'].str.len().values,
                'word_count': df['cleaned_text'].str.split().str.len().values
            }
        
        # 노트 특성
        note_columns = ['top_notes', 'middle_notes', 'base_notes']
        for col in note_columns:
            if col in df.columns:
                processed_notes = df[col].apply(self.process_notes)
                features[f'{col}_processed'] = processed_notes.tolist()
        
        # 수치형 특성
        numeric_columns = ['intensity', 'rating', 'longevity', 'sillage']
        numeric_features = []
        for col in numeric_columns:
            if col in df.columns:
                # 결측값 처리
                values = df[col].fillna(df[col].median()).values
                numeric_features.append(values)
        
        if numeric_features:
            numeric_array = np.column_stack(numeric_features)
            features['numeric'] = self.scaler.fit_transform(numeric_array)
            features['numeric_columns'] = [col for col in numeric_columns if col in df.columns]
        
        return features
    
    def create_training_dataset(
        self, 
        features: Dict[str, Any], 
        target_column: str = 'rating'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """훈련 데이터셋 생성"""
        # 모든 특성 결합
        feature_arrays = []
        
        if 'text_tfidf' in features:
            feature_arrays.append(features['text_tfidf'])
        
        if 'numeric' in features:
            feature_arrays.append(features['numeric'])
        
        # 텍스트 통계 추가
        if 'text_stats' in features:
            stats_array = np.column_stack([
                features['text_stats']['length'],
                features['text_stats']['word_count']
            ])
            feature_arrays.append(stats_array)
        
        if not feature_arrays:
            raise ValueError("추출된 특성이 없습니다.")
        
        X = np.hstack(feature_arrays)
        
        # 타겟 데이터 (예: 평점)
        if 'numeric' in features and target_column in features.get('numeric_columns', []):
            target_idx = features['numeric_columns'].index(target_column)
            y = features['numeric'][:, target_idx]
        else:
            # 기본값으로 랜덤 타겟 생성 (실제 사용시 실제 타겟 데이터 사용)
            y = np.random.rand(len(X))
        
        return X, y
    
    def save_processed_data(
        self, 
        features: Dict[str, Any], 
        output_path: str | Path
    ) -> None:
        """처리된 데이터 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 특성 데이터를 JSON으로 직렬화 가능한 형태로 변환
        serializable_features = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                serializable_features[key] = value.tolist()
            else:
                serializable_features[key] = value
        
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(serializable_features, f, ensure_ascii=False, indent=2)
        
        print(f"처리된 데이터 저장 완료: {output_path}")
    
    def process_pipeline(
        self, 
        input_path: str | Path, 
        output_path: str | Path
    ) -> Dict[str, Any]:
        """전체 데이터 처리 파이프라인 실행"""
        print("데이터 처리 파이프라인 시작...")
        
        # 1. 데이터 로드
        print("1. 원시 데이터 로드 중...")
        df = self.load_raw_data(input_path)
        print(f"로드된 데이터 수: {len(df)}")
        
        # 2. 데이터 검증
        print("2. 데이터 유효성 검증 중...")
        df = self.validate_data(df)
        
        # 3. 특성 추출
        print("3. 특성 추출 중...")
        features = self.extract_features(df)
        
        # 4. 처리된 데이터 저장
        print("4. 처리된 데이터 저장 중...")
        self.save_processed_data(features, output_path)
        
        print("데이터 처리 파이프라인 완료!")
        
        return features


def main() -> None:
    """메인 실행 함수"""
    processor = PerfumeDataProcessor()
    
    # 예시 데이터 처리
    input_path = Path("data/raw/raw_perfume_data.csv")
    output_path = Path("data/processed/processed_features.json")
    
    if input_path.exists():
        features = processor.process_pipeline(input_path, output_path)
        print(f"추출된 특성 키: {list(features.keys())}")
    else:
        print(f"입력 파일을 찾을 수 없습니다: {input_path}")


if __name__ == "__main__":
    main()
