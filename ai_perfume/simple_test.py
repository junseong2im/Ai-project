#!/usr/bin/env python3
"""
간단한 시스템 테스트 스크립트
"""

import json
from pathlib import Path

def test_dataset():
    """데이터셋 확인"""
    data_dir = Path("data/datasets")
    
    print("=== 데이터셋 테스트 ===")
    
    # 파일 확인
    train_file = data_dir / "fragrance_train.json"
    val_file = data_dir / "fragrance_validation.json"
    test_file = data_dir / "fragrance_test.json"
    
    if not train_file.exists():
        print("[X] 훈련 데이터셋이 없습니다")
        return False
        
    # 훈련 데이터 로드
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    materials = train_data["materials"]
    print(f"[OK] 훈련 데이터: {len(materials):,}개 샘플")
    
    # 첫 번째 샘플 검사
    sample = materials[0]
    print(f"[OK] 샘플 구조 확인:")
    print(f"   - 이름: {sample['name']}")
    print(f"   - 향기 계열: {sample['family']}")
    print(f"   - 원료 타입: {'천연' if sample['is_natural'] else '합성'}")
    print(f"   - 휘발성: {sample['olfactory_properties']['volatility']}")
    print(f"   - 강도: {sample['olfactory_properties']['intensity']}")
    print(f"   - 지속성: {sample['olfactory_properties']['longevity_hours']}시간")
    
    return True

def test_model_directory():
    """모델 디렉토리 확인"""
    model_dir = Path("models/fragrance_dl_models")
    
    print("\n=== 모델 디렉토리 테스트 ===")
    
    if not model_dir.exists():
        print("[X] 모델 디렉토리가 없습니다")
        return False
    
    # 모델 파일들 확인
    model_files = list(model_dir.glob("*.pth"))
    if model_files:
        print(f"[OK] 모델 파일 {len(model_files)}개 발견:")
        for f in model_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name}: {size_mb:.1f}MB")
    else:
        print("[WAIT] 모델 파일이 아직 생성되지 않았습니다 (훈련 진행 중)")
    
    return True

def main():
    print("=== Movie Scent AI 시스템 테스트 ===\n")
    
    # 데이터셋 테스트
    dataset_ok = test_dataset()
    
    # 모델 디렉토리 테스트
    model_ok = test_model_directory()
    
    print("\n=== 테스트 결과 ===")
    if dataset_ok and model_ok:
        print("[SUCCESS] 모든 테스트 통과!")
        print("[INFO] 딥러닝 훈련이 진행 중이거나 완료되었습니다.")
    else:
        print("[FAIL] 일부 테스트 실패")

if __name__ == "__main__":
    main()