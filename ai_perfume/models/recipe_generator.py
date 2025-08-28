from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import Config

class RecipeGenerator(nn.Module):
    """향수 레시피 생성 모델"""
    
    def __init__(self) -> None:
        super().__init__()
        
        # 모든 노트 목록 생성
        self.all_notes = self._get_all_notes()
        self.n_notes = len(self.all_notes)
        
        # 레이어 정의
        self.text_encoder = nn.Linear(768, 512)  # BERT 임베딩 크기
        self.emotion_encoder = nn.Linear(len(Config.EMOTIONS), 256)
        
        # 노트 선택기
        self.note_selector = nn.Sequential(
            nn.Linear(512 + 256, 512),  # 텍스트 인코딩 후 + 감정
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_notes * 3)  # top, middle, base notes
        )
        
        # 강도 예측기
        self.intensity_predictor = nn.Sequential(
            nn.Linear(512 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # GPU 사용 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(
        self, 
        text_embeddings: torch.Tensor, 
        emotion_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """순전파 함수"""
        # 입력을 GPU로 이동
        text_embeddings = text_embeddings.to(self.device)
        emotion_scores = emotion_scores.to(self.device)
        
        # 텍스트 인코딩
        text_features = self.text_encoder(text_embeddings)
        
        # 감정 인코딩
        emotion_features = self.emotion_encoder(emotion_scores)
        
        # 특성 결합
        combined = torch.cat([text_features, emotion_features], dim=1)
        
        # 노트 선택
        note_logits = self.note_selector(combined)
        note_probs = F.softmax(note_logits.view(-1, 3, self.n_notes), dim=2)
        
        # 강도 예측
        intensity = self.intensity_predictor(combined) * 10
        
        return note_probs, intensity
    
    def generate_recipe(
        self, 
        text_embeddings: torch.Tensor, 
        emotion_scores: torch.Tensor
    ) -> Dict[str, Any]:
        """레시피 생성"""
        note_probs, intensity = self(text_embeddings, emotion_scores)
        
        # 노트 선택
        selected_notes = self._select_notes(note_probs[0])  # 배치의 첫 번째 항목
        
        return {
            'top_notes': selected_notes['top'],
            'middle_notes': selected_notes['middle'],
            'base_notes': selected_notes['base'],
            'intensity': float(intensity[0].cpu().item())
        }
    
    def _select_notes(self, note_probs: torch.Tensor) -> Dict[str, List[str]]:
        """확률 기반 노트 선택"""
        import numpy as np
        
        # CPU로 이동
        note_probs_np = note_probs.cpu().detach().numpy()
        
        selected: Dict[str, List[str]] = {
            'top': [],
            'middle': [],
            'base': []
        }
        
        # 각 카테고리별로 2-3개의 노트 선택
        for i, category in enumerate(['top', 'middle', 'base']):
            probs = note_probs_np[i]
            n_select = random.randint(2, 3)  # 2-3개 랜덤 선택
            
            # 확률에 기반한 노트 선택
            selected_indices = (-probs).argsort()[:n_select]
            selected[category] = [self.all_notes[idx] for idx in selected_indices]
        
        return selected
    
    def _get_all_notes(self) -> List[str]:
        """모든 노트 목록 생성"""
        notes: List[str] = []
        for category in Config.NOTE_CATEGORIES.values():
            notes.extend(category)
        return list(set(notes))  # 중복 제거
    
    def save(self, path: str) -> None:
        """모델 저장"""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        """모델 로드"""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval() 