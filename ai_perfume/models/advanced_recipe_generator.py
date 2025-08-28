from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import Config


class PositionalEncoding(nn.Module):
    """위치 인코딩 - Transformer에서 시퀀스 위치 정보 제공"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadCrossAttention(nn.Module):
    """다중 모달 크로스 어텐션 - 텍스트와 향료 특성 간 상호작용"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.query(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        attention = self._compute_attention(Q, K, V, mask)
        
        # Reshape and apply output layer
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.out(attention)
    
    def _compute_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)


class FragranceExpertModule(nn.Module):
    """향료 카테고리별 전문가 모듈 - Mixture of Experts (MoE) 구현"""
    
    def __init__(self, d_model: int, d_ff: int, category: str):
        super().__init__()
        self.category = category
        self.expert_network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert_network(x)


class MixtureOfExperts(nn.Module):
    """향료 전문가들의 혼합 모델"""
    
    def __init__(self, d_model: int, d_ff: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        
        # 전문가 네트워크들
        self.experts = nn.ModuleList([
            FragranceExpertModule(d_model, d_ff, f"expert_{i}")
            for i in range(n_experts)
        ])
        
        # 게이트 네트워크 (어떤 전문가를 사용할지 결정)
        self.gate = nn.Linear(d_model, n_experts)
        
        # Top-k 전문가 선택 (일반적으로 k=2)
        self.top_k = min(2, n_experts)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        
        # 게이트 점수 계산
        gate_scores = self.gate(x_flat)  # (batch_size * seq_len, n_experts)
        
        # Top-k 전문가 선택
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=1)
        topk_weights = F.softmax(topk_scores, dim=1)
        
        # 전문가 출력 계산
        expert_outputs = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            weights = topk_weights[:, i].unsqueeze(1)
            
            # 각 전문가별로 처리
            for expert_id in range(self.n_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    expert_outputs[mask] += weights[mask] * expert_output
        
        return expert_outputs.view(batch_size, seq_len, d_model)


class AdvancedRecipeGenerator(nn.Module):
    """최신 Transformer 기반 향수 레시피 생성 모델"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, n_layers: int = 6, 
                 n_experts: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 향료 데이터 설정
        self.all_notes = self._get_all_notes()
        self.n_notes = len(self.all_notes)
        self.note_to_idx = {note: idx for idx, note in enumerate(self.all_notes)}
        
        # 감정 카테고리 설정
        self.emotions = Config.EMOTIONS
        self.n_emotions = len(self.emotions)
        
        # 입력 임베딩 레이어들
        self.text_projection = nn.Linear(768, d_model)  # BERT -> Transformer 차원
        self.emotion_embedding = nn.Embedding(self.n_emotions, d_model)
        self.note_embedding = nn.Embedding(self.n_notes, d_model)
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # 크로스 어텐션 (텍스트-감정-향료 상호작용)
        self.cross_attention = MultiHeadCrossAttention(d_model, n_heads, dropout)
        
        # Transformer 인코더 스택
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, n_layers)
        
        # Mixture of Experts 레이어
        self.moe = MixtureOfExperts(d_model, d_model * 2, n_experts)
        
        # 출력 헤드들
        self.note_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.n_notes * 3)  # top, middle, base
        )
        
        self.intensity_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # 향료 조합 최적화를 위한 어텐션 가중치
        self.composition_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 레이어 정규화
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.to(self.device)
    
    def forward(self, text_embeddings: torch.Tensor, emotion_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """순전파 함수"""
        batch_size = text_embeddings.size(0)
        
        # 입력을 디바이스로 이동
        text_embeddings = text_embeddings.to(self.device)
        emotion_scores = emotion_scores.to(self.device)
        
        # 텍스트 임베딩을 Transformer 차원으로 변환
        text_features = self.text_projection(text_embeddings)  # (batch, d_model)
        
        # 감정 점수를 임베딩으로 변환
        emotion_indices = torch.argmax(emotion_scores, dim=-1)  # (batch,)
        emotion_features = self.emotion_embedding(emotion_indices)  # (batch, d_model)
        
        # 시퀀스 구성: [텍스트, 감정, 향료 컨텍스트]
        sequence = torch.stack([text_features, emotion_features], dim=1)  # (batch, 2, d_model)
        
        # 위치 인코딩 적용
        sequence = sequence.transpose(0, 1)  # (2, batch, d_model)
        sequence = self.pos_encoding(sequence)
        sequence = sequence.transpose(0, 1)  # (batch, 2, d_model)
        
        # Transformer 인코딩
        encoded = self.transformer(sequence)  # (batch, 2, d_model)
        
        # Mixture of Experts 적용
        moe_output = self.moe(encoded)  # (batch, 2, d_model)
        
        # 크로스 어텐션으로 텍스트-감정 상호작용 강화
        cross_attended, _ = self.cross_attention.forward(
            moe_output, moe_output, moe_output
        )
        
        # 레이어 정규화 및 잔차 연결
        enhanced_features = self.layer_norm(cross_attended + moe_output)
        
        # 글로벌 풀링 (평균)
        pooled_features = enhanced_features.mean(dim=1)  # (batch, d_model)
        
        # 향료 조합 어텐션
        composition_features, attention_weights = self.composition_attention(
            pooled_features.unsqueeze(1),  # query
            enhanced_features,             # key
            enhanced_features              # value
        )
        composition_features = composition_features.squeeze(1)
        
        # 출력 예측
        note_logits = self.note_classifier(composition_features)
        note_probs = F.softmax(note_logits.view(-1, 3, self.n_notes), dim=2)
        
        intensity = self.intensity_predictor(composition_features) * 10
        
        return note_probs, intensity
    
    def generate_recipe(self, text_embeddings: torch.Tensor, emotion_scores: torch.Tensor,
                       temperature: float = 0.8, top_k: int = 5) -> Dict[str, Any]:
        """고급 레시피 생성 (온도 스케일링과 top-k 샘플링 적용)"""
        self.eval()
        
        with torch.no_grad():
            note_probs, intensity = self(text_embeddings, emotion_scores)
            
            # 온도 스케일링 적용
            note_probs = note_probs / temperature
            note_probs = F.softmax(note_probs, dim=2)
            
            # 각 카테고리별 노트 선택
            selected_notes = self._select_notes_advanced(
                note_probs[0], top_k=top_k
            )
            
            # 향료 조합 점수 계산 (화학적 조화도)
            composition_score = self._calculate_composition_harmony(selected_notes)
            
            return {
                'top_notes': selected_notes['top'],
                'middle_notes': selected_notes['middle'], 
                'base_notes': selected_notes['base'],
                'intensity': float(intensity[0].cpu().item()),
                'composition_harmony': composition_score,
                'confidence_scores': {
                    'top': float(note_probs[0, 0].max().cpu()),
                    'middle': float(note_probs[0, 1].max().cpu()),
                    'base': float(note_probs[0, 2].max().cpu())
                }
            }
    
    def _select_notes_advanced(self, note_probs: torch.Tensor, top_k: int = 5) -> Dict[str, List[str]]:
        """고급 노트 선택 (다양성과 조화를 고려한 샘플링)"""
        note_probs_np = note_probs.cpu().detach().numpy()
        
        selected = {'top': [], 'middle': [], 'base': []}
        categories = ['top', 'middle', 'base']
        
        for i, category in enumerate(categories):
            probs = note_probs_np[i]
            
            # Top-k 노트 선택
            top_indices = (-probs).argsort()[:top_k]
            top_probs = probs[top_indices]
            
            # 확률적 샘플링 (다양성 확보)
            n_select = min(3, len(top_indices))
            
            # 소프트맥스로 정규화
            normalized_probs = F.softmax(torch.tensor(top_probs), dim=0).numpy()
            
            # 확률 기반 샘플링 (중복 없이)
            selected_indices = []
            remaining_indices = list(range(len(top_indices)))
            remaining_probs = normalized_probs.copy()
            
            for _ in range(n_select):
                if not remaining_indices:
                    break
                
                # 확률 정규화
                remaining_probs = remaining_probs / remaining_probs.sum()
                
                # 샘플링
                choice_idx = torch.multinomial(
                    torch.tensor(remaining_probs), 1
                ).item()
                
                actual_idx = remaining_indices[choice_idx]
                selected_indices.append(top_indices[actual_idx])
                
                # 선택된 인덱스 제거
                remaining_indices.pop(choice_idx)
                remaining_probs = torch.cat([
                    torch.tensor(remaining_probs[:choice_idx]),
                    torch.tensor(remaining_probs[choice_idx+1:])
                ]).numpy()
            
            selected[category] = [self.all_notes[idx] for idx in selected_indices]
        
        return selected
    
    def _calculate_composition_harmony(self, selected_notes: Dict[str, List[str]]) -> float:
        """향료 조합의 화학적 조화도 계산"""
        # 간단한 휴리스틱 기반 조화도 계산
        # 실제 구현에서는 분자 구조 데이터를 활용할 수 있음
        
        harmony_score = 0.5  # 기본 점수
        
        # 카테고리별 균형성 확인
        total_notes = sum(len(notes) for notes in selected_notes.values())
        if 6 <= total_notes <= 9:  # 적절한 노트 수
            harmony_score += 0.2
        
        # 향료 그룹 다양성 확인 (시트러스, 플로럴, 우디 등)
        all_notes = []
        for notes in selected_notes.values():
            all_notes.extend(notes)
        
        unique_families = set()
        for note in all_notes:
            if 'citrus' in note.lower():
                unique_families.add('citrus')
            elif any(x in note.lower() for x in ['floral', 'flower']):
                unique_families.add('floral')
            elif any(x in note.lower() for x in ['woody', 'wood']):
                unique_families.add('woody')
            elif any(x in note.lower() for x in ['spicy', 'spice']):
                unique_families.add('spicy')
            elif any(x in note.lower() for x in ['fresh', 'marine']):
                unique_families.add('fresh')
        
        # 향료 패밀리 다양성 보너스
        diversity_bonus = min(len(unique_families) * 0.1, 0.3)
        harmony_score += diversity_bonus
        
        return min(harmony_score, 1.0)
    
    def _get_all_notes(self) -> List[str]:
        """모든 노트 목록 생성"""
        notes: List[str] = []
        for category in Config.NOTE_CATEGORIES.values():
            notes.extend(category)
        return list(set(notes))  # 중복 제거
    
    def get_attention_weights(self, text_embeddings: torch.Tensor, 
                             emotion_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """어텐션 가중치 추출 (해석 가능성을 위해)"""
        self.eval()
        
        with torch.no_grad():
            # Forward pass with attention extraction
            text_embeddings = text_embeddings.to(self.device)
            emotion_scores = emotion_scores.to(self.device)
            
            text_features = self.text_projection(text_embeddings)
            emotion_indices = torch.argmax(emotion_scores, dim=-1)
            emotion_features = self.emotion_embedding(emotion_indices)
            
            sequence = torch.stack([text_features, emotion_features], dim=1)
            sequence = sequence.transpose(0, 1)
            sequence = self.pos_encoding(sequence)
            sequence = sequence.transpose(0, 1)
            
            # Transformer 어텐션 추출은 복잡하므로 간단히 구현
            encoded = self.transformer(sequence)
            
            # 조합 어텐션 가중치
            _, composition_weights = self.composition_attention(
                encoded.mean(dim=1, keepdim=True),
                encoded,
                encoded
            )
            
            return {
                'composition_attention': composition_weights.cpu(),
                'input_importance': {
                    'text': 0.6,  # 휴리스틱 값
                    'emotion': 0.4
                }
            }
    
    def save_model(self, path: str, metadata: Optional[Dict] = None) -> None:
        """향상된 모델 저장 (메타데이터 포함)"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_notes': self.n_notes,
                'n_emotions': self.n_emotions
            },
            'note_vocabulary': self.all_notes,
            'note_to_idx': self.note_to_idx
        }
        
        if metadata:
            save_dict['metadata'] = metadata
        
        torch.save(save_dict, path)
    
    def load_model(self, path: str) -> Dict:
        """향상된 모델 로딩"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()
        
        return checkpoint.get('metadata', {})