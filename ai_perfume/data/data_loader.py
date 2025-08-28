from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

class DataLoader:
    """데이터 로딩 및 저장 관리 클래스"""
    
    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.training_data_path = self.data_dir / "training_data.json"
        self.feedback_data_path = self.data_dir / "feedback_data.json"
        
    def load_training_data(self) -> List[Dict[str, Any]]:
        """학습 데이터 로드"""
        if self.training_data_path.exists():
            with self.training_data_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_training_data(self, data: List[Dict[str, Any]]) -> None:
        """학습 데이터 저장"""
        self.data_dir.mkdir(exist_ok=True)
        with self.training_data_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_training_example(
        self, 
        text: str, 
        recipe: Dict[str, Any], 
        feedback: Optional[Dict[str, Any]] = None
    ) -> None:
        """새로운 학습 예제 추가"""
        data = self.load_training_data()
        
        example = {
            'id': len(data) + 1,
            'text': text,
            'recipe': recipe,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }
        
        data.append(example)
        self.save_training_data(data)
    
    def load_feedback_data(self) -> List[Dict[str, Any]]:
        """피드백 데이터 로드"""
        if self.feedback_data_path.exists():
            with self.feedback_data_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_feedback_data(self, data: List[Dict[str, Any]]) -> None:
        """피드백 데이터 저장"""
        self.data_dir.mkdir(exist_ok=True)
        with self.feedback_data_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_feedback(self, recipe_id: int, feedback: Dict[str, Any]) -> None:
        """새로운 피드백 추가"""
        data = self.load_feedback_data()
        
        feedback_entry = {
            'recipe_id': recipe_id,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }
        
        data.append(feedback_entry)
        self.save_feedback_data(data)
    
    def export_to_csv(self, output_dir: str | Path = "exports") -> None:
        """데이터를 CSV 형식으로 내보내기"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 학습 데이터 내보내기
        training_data = self.load_training_data()
        if training_data:
            df_training = pd.DataFrame(training_data)
            df_training.to_csv(
                output_path / "training_data.csv",
                index=False,
                encoding='utf-8-sig'
            )
        
        # 피드백 데이터 내보내기
        feedback_data = self.load_feedback_data()
        if feedback_data:
            df_feedback = pd.DataFrame(feedback_data)
            df_feedback.to_csv(
                output_path / "feedback_data.csv",
                index=False,
                encoding='utf-8-sig'
            ) 