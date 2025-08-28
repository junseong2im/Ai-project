# -*- coding: utf-8 -*-
import os
import pandas as pd
import re
from typing import List
import sys
import io

# 한글 출력을 위한 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    # 데이터 파일 경로
    raw_data_path = os.path.join('data', 'raw', 'raw_perfume_data.csv')
    
    # 원본 데이터 파일 존재 확인
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"원본 데이터 파일을 찾을 수 없습니다: {raw_data_path}")
    
    try:
        # 데이터 로드
        df = pd.read_csv(raw_data_path)
        print(f"\n데이터 로드 완료: {len(df)} 개의 레코드")
        print("컬럼 목록:", df.columns.tolist())
        
        # 컬럼 이름 매핑
        column_mapping = {
            'Name': 'name',
            'Gender': 'gender',
            'Rating Value': 'rating',
            'Rating Count': 'votes',
            'Main Accords': 'accords',
            'Perfumers': 'perfumers',
            'Description': 'description',
            'url': 'url'
        }
        
        # 컬럼 이름 변경
        df = df.rename(columns=column_mapping)
        print("\n매핑 후 컬럼:", df.columns.tolist())
        
        # 결측치 처리
        print("\n1. 결측치 처리 중...")
        essential_columns = ['name', 'description']
        df = df.dropna(subset=essential_columns)
        
        # 나머지 컬럼의 결측치는 적절한 값으로 대체
        df['rating'].fillna(0, inplace=True)
        df['votes'].fillna(0, inplace=True)
        df['gender'].fillna('Unisex', inplace=True)
        df['perfumers'].fillna('Unknown', inplace=True)
        df['accords'].fillna('', inplace=True)
        
        print(f"- 처리 후 레코드 수: {len(df)}")
        print("- 처리 후 컬럼:", df.columns.tolist())
        
        # 향료 정보 추출
        print("\n2. 향료 정보 추출 중...")
        df['notes'] = df['description'].apply(extract_notes)
        df['accords_list'] = df['accords'].apply(extract_accords)
        
        # 처리된 데이터 저장
        processed_dir = os.path.join('data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        output_path = os.path.join(processed_dir, 'processed_perfume_data.csv')
        df.to_csv(output_path, index=False)
        print(f"\n처리된 데이터 저장 완료: {output_path}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise

def extract_notes(text: str) -> List[str]:
    """설명에서 향료 정보 추출"""
    if pd.isna(text):
        return []
    
    # 향료 문자열 정규화
    text = str(text).lower()
    
    # 일반적인 향료 표현 패턴
    patterns = [
        r'notes?:?\s*([\w\s,]+)',  # "notes: lavender, rose"
        r'accords?:?\s*([\w\s,]+)',  # "accords: woody, floral"
        r'ingredients?:?\s*([\w\s,]+)',  # "ingredients: bergamot, jasmine"
        r'([\w\s]+)\s+notes?',  # "top notes", "base notes"
        r'scents? of\s+([\w\s,]+)'  # "scent of rose and jasmine"
    ]
    
    notes = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            notes.extend([n.strip() for n in match.split(',') if n.strip()])
    
    return list(set(notes))  # 중복 제거

def extract_accords(accords: str) -> List[str]:
    """Main Accords에서 향료 추출"""
    if pd.isna(accords):
        return []
    return [accord.strip() for accord in str(accords).split(',') if accord.strip()]

if __name__ == '__main__':
    main() 