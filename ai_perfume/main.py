from __future__ import annotations

import json
import random
from typing import Optional, Dict, Any, List

import torch

from models.text_analyzer import TextAnalyzer
from models.recipe_generator import RecipeGenerator
from core.learning_system import LearningSystem
from core.database import get_db_session, Recipe, Feedback, init_db
from config.settings import Config

class PerfumeAISystem:
    """AI 향수 시스템 메인 클래스"""
    
    def __init__(self) -> None:
        self.text_analyzer = TextAnalyzer()
        self.recipe_generator = RecipeGenerator()
        self.learning_system = LearningSystem(self.recipe_generator)
        
    def generate_recipe(self, text: str) -> Optional[Dict[str, Any]]:
        """향수 레시피 생성"""
        try:
            # 텍스트 분석
            analysis = self.text_analyzer.analyze(text)
            
            # 감정 점수를 텐서로 변환
            emotion_scores = torch.tensor(
                [list(analysis['emotions'].values())],
                dtype=torch.float32
            )
            
            # 레시피 생성
            recipe = self.recipe_generator.generate_recipe(
                analysis['embeddings'],
                emotion_scores
            )
            
            # 레시피 이름과 설명 생성
            recipe['name'] = self._generate_name(analysis)
            recipe['description'] = self._generate_description(text, analysis)
            
            # 데이터베이스에 저장 (컨텍스트 매니저 사용)
            with get_db_session() as session:
                db_recipe = Recipe(
                    name=recipe['name'],
                    description=recipe['description'],
                    top_notes=recipe['top_notes'],
                    middle_notes=recipe['middle_notes'],
                    base_notes=recipe['base_notes'],
                    intensity=recipe['intensity']
                )
                session.add(db_recipe)
                session.commit()
                recipe['id'] = db_recipe.id
            
            return recipe
            
        except Exception as e:
            print(f"레시피 생성 중 오류 발생: {e}")
            return None
    
    def process_feedback(self, recipe_id: int, feedback_data: Dict[str, Any]) -> bool:
        """피드백 처리 및 학습"""
        try:
            # 피드백 저장 (컨텍스트 매니저 사용)
            with get_db_session() as session:
                db_feedback = Feedback(
                    recipe_id=recipe_id,
                    rating=feedback_data['rating'],
                    comments=feedback_data.get('comments', ''),
                    emotional_response=feedback_data['emotional_response'],
                    longevity=feedback_data['longevity'],
                    sillage=feedback_data['sillage']
                )
                session.add(db_feedback)
                session.commit()
            
            # 충분한 피드백이 모이면 학습 실행
            if self._should_train():
                loss = self.learning_system.train()
                print(f"모델 학습 완료 (Loss: {loss:.4f})")
                
            return True
            
        except Exception as e:
            print(f"피드백 처리 중 오류 발생: {e}")
            return False
    
    def _generate_name(self, analysis: Dict[str, Any]) -> str:
        """향수 이름 생성"""
        # 주요 감정과 키워드 추출
        top_emotion = max(analysis['emotions'].items(), key=lambda x: x[1])[0]
        keywords = analysis['keywords']
        
        if not keywords:
            return f"{top_emotion}의 향기"
        
        # 이름 템플릿
        templates = [
            f"{keywords[0]}의 {top_emotion}",
            f"{top_emotion}한 {keywords[0]}",
            f"{keywords[0]} 속 {top_emotion}"
        ]
        
        return random.choice(templates)
    
    def _generate_description(self, text: str, analysis: Dict[str, Any]) -> str:
        """향수 설명 생성"""
        top_emotion = max(analysis['emotions'].items(), key=lambda x: x[1])[0]
        
        description = f"{text[:50]}{'...' if len(text) > 50 else ''} 의 순간을 담은 향수입니다. "
        description += f"{top_emotion}의 감성을 향으로 표현했습니다."
        
        return description
    
    def _should_train(self) -> bool:
        """학습 실행 여부 결정"""
        with get_db_session() as session:
            feedback_count = session.query(Feedback).count()
            return feedback_count >= Config.MIN_FEEDBACK_COUNT
    
    def save_model(self, path: str) -> None:
        """모델 저장"""
        self.recipe_generator.save(path)
    
    def load_model(self, path: str) -> None:
        """모델 로드"""
        self.recipe_generator.load(path)


def main() -> None:
    """메인 실행 함수"""
    # 데이터베이스 초기화
    init_db()
    
    # AI 시스템 초기화
    perfume_system = PerfumeAISystem()
    
    # 테스트 실행
    test_texts = [
        "여름 저녁 해변에서 불어오는 시원한 바닷바람",
        "비 오는 가을 오후, 창가에서 커피를 마시며",
        "봄날 아침 공원에서 피어나는 꽃들의 향기",
        "겨울 밤 벽난로 앞에서 읽는 책의 분위기"
    ]
    
    for text in test_texts:
        print(f"\n입력 텍스트: {text}")
        recipe = perfume_system.generate_recipe(text)
        
        if recipe:
            print(f"향수 이름: {recipe['name']}")
            print(f"설명: {recipe['description']}")
            print("레시피:")
            print(f"- 탑 노트: {', '.join(recipe['top_notes'])}")
            print(f"- 미들 노트: {', '.join(recipe['middle_notes'])}")
            print(f"- 베이스 노트: {', '.join(recipe['base_notes'])}")
            print(f"강도: {recipe['intensity']}/10")
            
            # 테스트 피드백
            feedback: Dict[str, Any] = {
                'rating': 4.5,
                'comments': "좋은 향입니다",
                'emotional_response': ['refreshing', 'peaceful'],
                'longevity': 3,
                'sillage': 4
            }
            
            perfume_system.process_feedback(recipe['id'], feedback)

if __name__ == "__main__":
    main() 