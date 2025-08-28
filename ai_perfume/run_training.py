#!/usr/bin/env python3
"""
향료 원료 딥러닝 훈련 실행 스크립트
200,000개 데이터셋으로 편향 없는 AI 모델 구축
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from training.deep_learning_trainer import FragranceTrainer

def main():
    """메인 훈련 함수"""
    print("=== 향료 원료 딥러닝 훈련 시스템 ===")
    print("200,000개 데이터셋으로 편향 없는 AI 모델 구축\n")
    
    # 데이터 경로 설정
    data_dir = current_dir / "data" / "datasets"
    model_save_dir = current_dir / "models" / "fragrance_dl_models"
    
    # 디렉토리 존재 확인
    if not data_dir.exists():
        print(f"ERROR: 데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        print("먼저 data_generation/fragrance_dataset_generator.py를 실행하여 데이터셋을 생성하세요.")
        return
    
    # 필요한 파일들 확인
    required_files = [
        data_dir / "fragrance_train.json",
        data_dir / "fragrance_validation.json", 
        data_dir / "fragrance_test.json"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("ERROR: 필요한 데이터 파일이 없습니다:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n먼저 data_generation/fragrance_dataset_generator.py를 실행하여 데이터셋을 생성하세요.")
        return
    
    print(f"데이터 디렉토리: {data_dir}")
    print(f"모델 저장 디렉토리: {model_save_dir}")
    
    try:
        # 훈련기 초기화
        print("\n=== 모델 초기화 중... ===")
        trainer = FragranceTrainer(str(data_dir), str(model_save_dir))
        
        # 모델 훈련
        print("\n=== 훈련 시작 ===")
        trainer.train(epochs=50, early_stopping_patience=15)
        
        print("\n=== 훈련 완료! ===")
        print(f"최고 모델 저장 위치: {trainer.model_save_dir}")
        print(f"최고 검증 손실: {trainer.best_val_loss:.6f}")
        print("\n모델 훈련이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"\nERROR: 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()