from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Warning: Gymnasium not available. RL environment will be simulated.")


@dataclass
class UserFeedback:
    """사용자 피드백 구조"""
    recipe_id: str
    overall_rating: float  # 1-10 scale
    fragrance_notes: Dict[str, float]  # note별 만족도
    emotional_response: List[str]  # ['happy', 'calm', 'energetic', etc.]
    longevity_rating: float  # 1-10 scale
    sillage_rating: float  # 1-10 scale
    occasion_appropriateness: Dict[str, float]  # occasion별 적합성
    improvement_suggestions: List[str]
    user_id: str
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FragranceAction:
    """향수 레시피 액션"""
    top_notes: List[str]
    middle_notes: List[str]
    base_notes: List[str]
    intensity: float
    note_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class FragranceState:
    """향수 추천 환경의 상태"""
    user_preferences: Dict[str, Any]
    context: Dict[str, Any]  # season, time, occasion, etc.
    emotion_target: List[str]
    previous_feedback: List[UserFeedback]
    user_history: Dict[str, Any]


class FragranceRewardCalculator:
    """향수 피드백 기반 보상 계산기"""
    
    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        self.reward_weights = reward_weights or {
            'overall_satisfaction': 0.3,
            'emotional_match': 0.25,
            'longevity': 0.15,
            'sillage': 0.15,
            'innovation_bonus': 0.10,
            'context_appropriateness': 0.05
        }
    
    def calculate_reward(
        self, 
        action: FragranceAction, 
        feedback: UserFeedback, 
        state: FragranceState
    ) -> Tuple[float, Dict[str, float]]:
        """피드백을 기반으로 보상 계산"""
        
        reward_components = {}
        
        # 1. 전체 만족도 보상
        overall_reward = (feedback.overall_rating - 5) / 5  # -1 to 1 scale
        reward_components['overall_satisfaction'] = overall_reward
        
        # 2. 감정 매칭 보상
        emotion_reward = self._calculate_emotion_reward(
            action, feedback.emotional_response, state.emotion_target
        )
        reward_components['emotional_match'] = emotion_reward
        
        # 3. 지속성 보상
        longevity_reward = (feedback.longevity_rating - 5) / 5
        reward_components['longevity'] = longevity_reward
        
        # 4. 확산력 보상
        sillage_reward = (feedback.sillage_rating - 5) / 5
        reward_components['sillage'] = sillage_reward
        
        # 5. 혁신성 보너스 (새로운 조합 시도)
        innovation_reward = self._calculate_innovation_reward(action, state)
        reward_components['innovation_bonus'] = innovation_reward
        
        # 6. 상황 적합성 보상
        context_reward = self._calculate_context_reward(
            feedback.occasion_appropriateness, state.context
        )
        reward_components['context_appropriateness'] = context_reward
        
        # 가중 합계
        total_reward = sum(
            self.reward_weights[key] * value
            for key, value in reward_components.items()
            if key in self.reward_weights
        )
        
        return total_reward, reward_components
    
    def _calculate_emotion_reward(
        self, 
        action: FragranceAction, 
        actual_emotions: List[str], 
        target_emotions: List[str]
    ) -> float:
        """감정 타겟과 실제 반응의 매칭 보상"""
        if not target_emotions or not actual_emotions:
            return 0.0
        
        # 감정 매칭 점수
        matches = len(set(actual_emotions) & set(target_emotions))
        total_targets = len(target_emotions)
        
        match_ratio = matches / total_targets if total_targets > 0 else 0
        
        # -1 to 1 스케일로 변환
        return (match_ratio - 0.5) * 2
    
    def _calculate_innovation_reward(
        self, 
        action: FragranceAction, 
        state: FragranceState
    ) -> float:
        """혁신적인 조합에 대한 보너스"""
        all_notes = action.top_notes + action.middle_notes + action.base_notes
        
        # 과거 피드백에서 사용된 노트들과의 중복도 확인
        if not state.previous_feedback:
            return 0.5  # 첫 추천에는 중간 점수
        
        used_notes = set()
        for feedback in state.previous_feedback[-10:]:  # 최근 10개 피드백만 확인
            # 피드백에서 노트 정보 추출 (간단화)
            for note_rating in feedback.fragrance_notes.keys():
                used_notes.add(note_rating)
        
        current_notes = set(all_notes)
        new_notes_ratio = len(current_notes - used_notes) / len(current_notes) if current_notes else 0
        
        # 새로운 노트 비율에 따른 혁신성 보상
        return (new_notes_ratio - 0.3) * 2  # 30% 이상 새로운 노트면 플러스
    
    def _calculate_context_reward(
        self, 
        occasion_ratings: Dict[str, float], 
        context: Dict[str, Any]
    ) -> float:
        """상황별 적합성 보상"""
        if not occasion_ratings or not context:
            return 0.0
        
        target_occasion = context.get('occasion', 'casual')
        
        if target_occasion in occasion_ratings:
            rating = occasion_ratings[target_occasion]
            return (rating - 5) / 5  # 1-10을 -1 to 1로 변환
        
        return 0.0


class PPOFragrancePolicy(nn.Module):
    """PPO 기반 향수 추천 정책 네트워크"""
    
    def __init__(
        self,
        state_dim: int,
        action_space_size: int,
        hidden_dim: int = 256,
        n_layers: int = 3
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        
        # 정책 네트워크 (Actor)
        policy_layers = []
        current_dim = state_dim
        
        for _ in range(n_layers):
            policy_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        policy_layers.append(nn.Linear(hidden_dim, action_space_size))
        self.policy_network = nn.Sequential(*policy_layers)
        
        # 가치 네트워크 (Critic)
        value_layers = []
        current_dim = state_dim
        
        for _ in range(n_layers):
            value_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        value_layers.append(nn.Linear(hidden_dim, 1))
        self.value_network = nn.Sequential(*value_layers)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """정책과 가치 동시 출력"""
        policy_logits = self.policy_network(state)
        value = self.value_network(state)
        
        return policy_logits, value.squeeze(-1)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """상태에서 액션 샘플링"""
        policy_logits, value = self.forward(state)
        action_probs = F.softmax(policy_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1))).squeeze(-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """액션 평가 (학습용)"""
        policy_logits, values = self.forward(states)
        action_probs = F.softmax(policy_logits, dim=-1)
        dist = Categorical(action_probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return values, log_probs, entropy


class DPOFragranceTrainer:
    """Direct Preference Optimization 향수 추천 트레이너"""
    
    def __init__(
        self,
        policy_model: PPOFragrancePolicy,
        beta: float = 0.1,  # DPO 정규화 파라미터
        learning_rate: float = 1e-4
    ):
        self.policy_model = policy_model
        self.reference_model = PPOFragrancePolicy(
            policy_model.state_dim,
            policy_model.action_space_size
        )
        
        # 레퍼런스 모델을 현재 정책으로 초기화
        self.reference_model.load_state_dict(policy_model.state_dict())
        self.reference_model.eval()
        
        self.beta = beta
        self.optimizer = optim.AdamW(policy_model.parameters(), lr=learning_rate)
        
        self.preference_data: List[Dict[str, Any]] = []
    
    def add_preference_pair(
        self,
        state: torch.Tensor,
        chosen_action: FragranceAction,
        rejected_action: FragranceAction,
        preference_strength: float = 1.0
    ):
        """선호도 쌍 데이터 추가"""
        self.preference_data.append({
            'state': state,
            'chosen_action': chosen_action,
            'rejected_action': rejected_action,
            'preference_strength': preference_strength
        })
    
    def compute_dpo_loss(self, batch_size: int = 32) -> torch.Tensor:
        """DPO 손실 함수 계산"""
        if len(self.preference_data) < batch_size:
            return torch.tensor(0.0)
        
        # 배치 샘플링
        batch_indices = random.sample(range(len(self.preference_data)), batch_size)
        batch_data = [self.preference_data[i] for i in batch_indices]
        
        total_loss = 0.0
        
        for data in batch_data:
            state = data['state']
            chosen_action = data['chosen_action']
            rejected_action = data['rejected_action']
            
            # 정책 로그 확률 계산
            policy_logits_chosen, _ = self.policy_model(state)
            policy_logits_rejected, _ = self.policy_model(state)
            
            # 레퍼런스 모델 로그 확률
            with torch.no_grad():
                ref_logits_chosen, _ = self.reference_model(state)
                ref_logits_rejected, _ = self.reference_model(state)
            
            # 액션 인덱스 변환 (간단화)
            chosen_idx = self._action_to_index(chosen_action)
            rejected_idx = self._action_to_index(rejected_action)
            
            # DPO 손실 계산
            policy_log_ratio = (
                F.log_softmax(policy_logits_chosen, dim=-1)[chosen_idx] -
                F.log_softmax(policy_logits_rejected, dim=-1)[rejected_idx]
            )
            
            ref_log_ratio = (
                F.log_softmax(ref_logits_chosen, dim=-1)[chosen_idx] -
                F.log_softmax(ref_logits_rejected, dim=-1)[rejected_idx]
            )
            
            loss = -F.logsigmoid(self.beta * (policy_log_ratio - ref_log_ratio))
            total_loss += loss
        
        return total_loss / batch_size
    
    def _action_to_index(self, action: FragranceAction) -> int:
        """액션을 인덱스로 변환 (간단화)"""
        # 실제 구현에서는 더 복잡한 액션 공간 매핑 필요
        all_notes = action.top_notes + action.middle_notes + action.base_notes
        return hash(tuple(sorted(all_notes))) % self.policy_model.action_space_size
    
    def update_policy(self) -> float:
        """정책 업데이트"""
        loss = self.compute_dpo_loss()
        
        if loss.item() > 0:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
        
        return loss.item()


class FragranceRLSystem:
    """향수 추천 강화학습 통합 시스템"""
    
    def __init__(
        self,
        state_encoder_dim: int = 512,
        action_space_size: int = 1000,  # 향수 조합 수
        use_dpo: bool = True
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 정책 네트워크 초기화
        self.policy = PPOFragrancePolicy(
            state_dim=state_encoder_dim,
            action_space_size=action_space_size
        ).to(self.device)
        
        # DPO 트레이너 (선택적)
        if use_dpo:
            self.dpo_trainer = DPOFragranceTrainer(self.policy)
        else:
            self.dpo_trainer = None
        
        # 보상 계산기
        self.reward_calculator = FragranceRewardCalculator()
        
        # 경험 메모리
        self.experience_buffer: deque = deque(maxlen=10000)
        self.user_feedback_history: Dict[str, List[UserFeedback]] = defaultdict(list)
        
        # 학습 통계
        self.training_stats = {
            'total_episodes': 0,
            'total_feedback': 0,
            'average_reward': 0.0,
            'policy_updates': 0
        }
    
    def encode_state(self, state: FragranceState) -> torch.Tensor:
        """상태를 벡터로 인코딩"""
        # 간단한 상태 인코딩 (실제로는 더 복잡한 인코더 사용)
        encoded_features = []
        
        # 사용자 선호도 인코딩
        prefs = state.user_preferences
        encoded_features.extend([
            prefs.get('intensity_preference', 5.0) / 10.0,
            prefs.get('longevity_preference', 5.0) / 10.0,
            prefs.get('sillage_preference', 5.0) / 10.0,
        ])
        
        # 컨텍스트 인코딩
        context = state.context
        season_encoding = [0, 0, 0, 0]  # spring, summer, autumn, winter
        if context.get('season') in ['spring', 'summer', 'autumn', 'winter']:
            season_idx = ['spring', 'summer', 'autumn', 'winter'].index(context['season'])
            season_encoding[season_idx] = 1
        encoded_features.extend(season_encoding)
        
        # 감정 타겟 인코딩
        emotion_encoding = [0] * 8  # 8개 주요 감정
        emotions = ['joy', 'calm', 'energetic', 'romantic', 'mysterious', 'fresh', 'warm', 'sophisticated']
        for emotion in state.emotion_target:
            if emotion in emotions:
                emotion_idx = emotions.index(emotion)
                emotion_encoding[emotion_idx] = 1
        encoded_features.extend(emotion_encoding)
        
        # 피드백 히스토리 요약
        if state.previous_feedback:
            recent_feedback = state.previous_feedback[-5:]  # 최근 5개
            avg_rating = sum(fb.overall_rating for fb in recent_feedback) / len(recent_feedback)
            avg_longevity = sum(fb.longevity_rating for fb in recent_feedback) / len(recent_feedback)
            avg_sillage = sum(fb.sillage_rating for fb in recent_feedback) / len(recent_feedback)
            
            encoded_features.extend([
                avg_rating / 10.0,
                avg_longevity / 10.0,
                avg_sillage / 10.0
            ])
        else:
            encoded_features.extend([0.5, 0.5, 0.5])
        
        # 패딩으로 고정 크기 맞추기
        while len(encoded_features) < 512:
            encoded_features.append(0.0)
        
        return torch.tensor(encoded_features[:512], dtype=torch.float32).to(self.device)
    
    def get_recommendation(
        self, 
        state: FragranceState, 
        deterministic: bool = False
    ) -> Tuple[FragranceAction, float]:
        """상태에 기반한 향수 추천"""
        self.policy.eval()
        
        encoded_state = self.encode_state(state).unsqueeze(0)
        
        with torch.no_grad():
            action_idx, log_prob = self.policy.get_action(encoded_state, deterministic)
            _, value = self.policy(encoded_state)
        
        # 액션 인덱스를 실제 향수 레시피로 변환
        fragrance_action = self._index_to_action(action_idx.item(), state)
        
        return fragrance_action, value.item()
    
    def _index_to_action(self, action_idx: int, state: FragranceState) -> FragranceAction:
        """액션 인덱스를 향수 레시피로 변환"""
        # 사전 정의된 향수 조합들 (실제로는 더 정교한 매핑 필요)
        available_notes = {
            'top': ['bergamot', 'lemon', 'grapefruit', 'mint', 'green', 'marine'],
            'middle': ['rose', 'jasmine', 'lavender', 'floral', 'spicy', 'fruity'], 
            'base': ['vanilla', 'musk', 'amber', 'woody', 'sandalwood', 'cedar']
        }
        
        # 의사 랜덤 선택 (액션 인덱스 기반)
        random.seed(action_idx)
        
        # 사용자 선호도와 컨텍스트 고려한 선택
        top_notes = random.sample(available_notes['top'], k=random.randint(2, 3))
        middle_notes = random.sample(available_notes['middle'], k=random.randint(2, 3))
        base_notes = random.sample(available_notes['base'], k=random.randint(1, 3))
        
        # 강도는 사용자 선호도 기반
        intensity_pref = state.user_preferences.get('intensity_preference', 5.0)
        intensity = max(1.0, min(10.0, intensity_pref + random.uniform(-2, 2)))
        
        return FragranceAction(
            top_notes=top_notes,
            middle_notes=middle_notes,
            base_notes=base_notes,
            intensity=intensity
        )
    
    def process_feedback(
        self,
        state: FragranceState,
        action: FragranceAction,
        feedback: UserFeedback
    ) -> Dict[str, Any]:
        """피드백 처리 및 학습"""
        # 보상 계산
        reward, reward_components = self.reward_calculator.calculate_reward(
            action, feedback, state
        )
        
        # 경험 저장
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'feedback': feedback,
            'reward_components': reward_components
        }
        self.experience_buffer.append(experience)
        
        # 사용자별 피드백 히스토리 저장
        self.user_feedback_history[feedback.user_id].append(feedback)
        
        # DPO 학습 데이터 생성 (선호도 기반)
        if self.dpo_trainer and len(self.experience_buffer) >= 2:
            self._generate_preference_pairs(experience)
        
        # 학습 수행 (충분한 데이터가 모이면)
        training_result = {}
        if len(self.experience_buffer) >= 100:  # 최소 학습 데이터 수
            training_result = self._perform_training()
        
        # 통계 업데이트
        self.training_stats['total_feedback'] += 1
        self.training_stats['average_reward'] = (
            self.training_stats['average_reward'] * (self.training_stats['total_feedback'] - 1) + reward
        ) / self.training_stats['total_feedback']
        
        return {
            'reward': reward,
            'reward_components': reward_components,
            'training_result': training_result,
            'experience_buffer_size': len(self.experience_buffer)
        }
    
    def _generate_preference_pairs(self, current_experience: Dict[str, Any]):
        """현재 경험을 기반으로 선호도 쌍 생성"""
        current_reward = current_experience['reward']
        current_state = current_experience['state']
        current_action = current_experience['action']
        
        # 과거 경험 중 같은 사용자의 경험 찾기
        user_id = current_experience['feedback'].user_id
        user_experiences = [
            exp for exp in self.experience_buffer
            if exp['feedback'].user_id == user_id
        ]
        
        if len(user_experiences) < 2:
            return
        
        # 보상이 더 낮은 경험과 쌍 만들기
        for past_exp in user_experiences[-5:]:  # 최근 5개만 확인
            past_reward = past_exp['reward']
            
            if current_reward > past_reward + 0.1:  # 충분한 차이가 있을 때만
                state_tensor = self.encode_state(current_state)
                
                self.dpo_trainer.add_preference_pair(
                    state=state_tensor,
                    chosen_action=current_action,
                    rejected_action=past_exp['action'],
                    preference_strength=abs(current_reward - past_reward)
                )
    
    def _perform_training(self) -> Dict[str, Any]:
        """정책 네트워크 학습"""
        self.policy.train()
        
        training_result = {'policy_loss': 0.0, 'dpo_loss': 0.0}
        
        # DPO 학습
        if self.dpo_trainer:
            dpo_loss = self.dpo_trainer.update_policy()
            training_result['dpo_loss'] = dpo_loss
            self.training_stats['policy_updates'] += 1
        
        return training_result
    
    def get_user_personalization(self, user_id: str) -> Dict[str, Any]:
        """사용자별 개인화 정보"""
        user_feedback = self.user_feedback_history.get(user_id, [])
        
        if not user_feedback:
            return {'personalization_level': 'none', 'recommendations': []}
        
        # 선호도 패턴 분석
        avg_ratings = {
            'overall': sum(fb.overall_rating for fb in user_feedback) / len(user_feedback),
            'longevity': sum(fb.longevity_rating for fb in user_feedback) / len(user_feedback),
            'sillage': sum(fb.sillage_rating for fb in user_feedback) / len(user_feedback)
        }
        
        # 선호하는 감정 반응
        all_emotions = []
        for fb in user_feedback:
            all_emotions.extend(fb.emotional_response)
        
        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        preferred_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'personalization_level': 'high' if len(user_feedback) >= 10 else 'medium' if len(user_feedback) >= 5 else 'low',
            'feedback_count': len(user_feedback),
            'average_ratings': avg_ratings,
            'preferred_emotions': [emotion for emotion, _ in preferred_emotions],
            'satisfaction_trend': self._calculate_satisfaction_trend(user_feedback)
        }
    
    def _calculate_satisfaction_trend(self, feedback_list: List[UserFeedback]) -> str:
        """만족도 트렌드 분석"""
        if len(feedback_list) < 3:
            return 'insufficient_data'
        
        recent_avg = sum(fb.overall_rating for fb in feedback_list[-3:]) / 3
        older_avg = sum(fb.overall_rating for fb in feedback_list[:-3]) / max(1, len(feedback_list) - 3)
        
        if recent_avg > older_avg + 0.5:
            return 'improving'
        elif recent_avg < older_avg - 0.5:
            return 'declining'
        else:
            return 'stable'
    
    def save_model(self, path: str):
        """모델과 학습 데이터 저장"""
        save_data = {
            'policy_state_dict': self.policy.state_dict(),
            'training_stats': self.training_stats,
            'experience_buffer': list(self.experience_buffer)[-1000:],  # 최근 1000개만
            'user_feedback_history': dict(self.user_feedback_history)
        }
        
        torch.save(save_data, path)
    
    def load_model(self, path: str):
        """모델과 학습 데이터 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        if 'experience_buffer' in checkpoint:
            self.experience_buffer.extend(checkpoint['experience_buffer'])
        
        if 'user_feedback_history' in checkpoint:
            for user_id, feedback_list in checkpoint['user_feedback_history'].items():
                self.user_feedback_history[user_id] = feedback_list
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """시스템 성능 메트릭"""
        metrics = {
            'training_stats': self.training_stats.copy(),
            'experience_buffer_size': len(self.experience_buffer),
            'unique_users': len(self.user_feedback_history),
            'total_feedback': sum(len(fb_list) for fb_list in self.user_feedback_history.values()),
            'device': str(self.device),
            'policy_parameters': sum(p.numel() for p in self.policy.parameters())
        }
        
        # 최근 성과 분석
        if len(self.experience_buffer) >= 10:
            recent_rewards = [exp['reward'] for exp in list(self.experience_buffer)[-10:]]
            metrics['recent_average_reward'] = sum(recent_rewards) / len(recent_rewards)
            metrics['reward_variance'] = np.var(recent_rewards)
        
        return metrics