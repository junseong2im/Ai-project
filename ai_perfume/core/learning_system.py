from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import Config
from core.database import get_db_session, Recipe, Feedback

class PerfumeDataset(Dataset[Dict[str, Any]]):
    """향수 데이터셋 클래스"""
    
    def __init__(self, recipes: List[Recipe], feedbacks: List[Feedback]) -> None:
        self.recipes = recipes
        self.feedbacks = feedbacks
        
    def __len__(self) -> int:
        return len(self.recipes)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        recipe = self.recipes[idx]
        feedback = self.feedbacks[idx]
        
        # 레시피 데이터 준비
        recipe_data = {
            'top_notes': recipe.top_notes,
            'middle_notes': recipe.middle_notes,
            'base_notes': recipe.base_notes,
            'intensity': recipe.intensity
        }
        
        # 피드백 데이터 준비
        feedback_data = {
            'rating': feedback.rating,
            'emotional_response': feedback.emotional_response,
            'longevity': feedback.longevity,
            'sillage': feedback.sillage
        }
        
        return {
            'recipe': recipe_data,
            'feedback': feedback_data
        }

class LearningSystem:
    """학습 시스템 클래스"""
    
    def __init__(self, model: Any, optimizer: Optional[Any] = None) -> None:
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE
        )
        
    def train(self, batch_size: int = Config.BATCH_SIZE) -> Optional[float]:
        """모델 학습 실행"""
        # 학습 데이터 수집
        recipes, feedbacks = self._collect_training_data()
        if not recipes:
            print("학습할 데이터가 없습니다.")
            return None
            
        # 데이터셋 생성
        dataset = PerfumeDataset(recipes, feedbacks)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # 학습 실행
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # 레시피 생성
            recipe_data = batch['recipe']
            feedback_data = batch['feedback']
            
            # 손실 계산
            loss = self._calculate_loss(recipe_data, feedback_data)
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 평균 손실 반환
        avg_loss = total_loss / len(dataloader)
        
        # 학습 결과 저장
        self._save_training_result(avg_loss)
        
        return avg_loss
    
    def _collect_training_data(self) -> Tuple[List[Recipe], List[Feedback]]:
        """학습 데이터 수집"""
        with get_db_session() as session:
            # 피드백이 있는 레시피만 선택
            feedbacks = list(session.query(Feedback).all())
            recipe_ids = [f.recipe_id for f in feedbacks]
            recipes = list(session.query(Recipe).filter(Recipe.id.in_(recipe_ids)).all())
        
        return recipes, feedbacks
    
    def _calculate_loss(
        self, 
        recipe_data: Dict[str, Any], 
        feedback_data: Dict[str, Any]
    ) -> torch.Tensor:
        """손실 함수 계산"""
        # 노트 선택 손실
        note_loss = self._calculate_note_loss(
            recipe_data['top_notes'],
            recipe_data['middle_notes'],
            recipe_data['base_notes']
        )
        
        # 강도 예측 손실
        intensity_loss = F.mse_loss(
            torch.tensor([recipe_data['intensity']], dtype=torch.float32),
            torch.tensor([feedback_data['rating']], dtype=torch.float32)
        )
        
        # 전체 손실
        total_loss = note_loss + Config.FEEDBACK_WEIGHT * intensity_loss
        
        return total_loss
    
    def _calculate_note_loss(
        self, 
        top_notes: List[str], 
        middle_notes: List[str], 
        base_notes: List[str]
    ) -> torch.Tensor:
        """노트 선택에 대한 손실 계산"""
        # 간단한 더미 손실 (실제 구현에서는 더 복잡한 로직 필요)
        note_count = len(top_notes) + len(middle_notes) + len(base_notes)
        return torch.tensor(0.1 * note_count, dtype=torch.float32)
    
    def _save_training_result(self, loss: float) -> None:
        """학습 결과 저장"""
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'loss': loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # 결과를 파일에 저장
        history_path = Path('training_history.json')
        with history_path.open('a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    def save_model(self, path: str) -> None:
        """모델 저장"""
        self.model.save(path)
    
    def load_model(self, path: str) -> None:
        """모델 로드"""
        self.model.load(path) 