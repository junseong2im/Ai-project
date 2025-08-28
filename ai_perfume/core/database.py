from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, String, Float, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session as SQLSession
from sqlalchemy.sql import func

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import Config


class Base(DeclarativeBase):
    """SQLAlchemy 2.0+ 스타일 베이스 클래스"""
    pass


# 엔진 생성 (SQLAlchemy 2.0+ 스타일)
engine = create_engine(
    Config.DATABASE_URL,
    echo=False,  # 개발 시에는 True로 설정
    future=True  # SQLAlchemy 2.0 스타일 사용
)

# 세션 팩토리 생성
SessionLocal = sessionmaker(
    bind=engine,
    class_=SQLSession,
    expire_on_commit=False
)

class Recipe(Base):
    """향수 레시피 모델"""
    __tablename__ = 'recipes'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(String(1000))
    top_notes: Mapped[List[str]] = mapped_column(JSON)
    middle_notes: Mapped[List[str]] = mapped_column(JSON)
    base_notes: Mapped[List[str]] = mapped_column(JSON)
    intensity: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now()
    )
    
    def __repr__(self) -> str:
        return f"<Recipe(id={self.id}, name='{self.name}', intensity={self.intensity})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """모델을 딕셔너리로 변환"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'top_notes': self.top_notes,
            'middle_notes': self.middle_notes,
            'base_notes': self.base_notes,
            'intensity': self.intensity,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Feedback(Base):
    """피드백 모델"""
    __tablename__ = 'feedback'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    recipe_id: Mapped[int] = mapped_column()  # 외래 키는 별도로 정의 가능
    rating: Mapped[float] = mapped_column(Float)
    comments: Mapped[Optional[str]] = mapped_column(String(1000))
    emotional_response: Mapped[List[str]] = mapped_column(JSON)
    longevity: Mapped[int] = mapped_column()
    sillage: Mapped[int] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now()
    )
    
    def __repr__(self) -> str:
        return f"<Feedback(id={self.id}, recipe_id={self.recipe_id}, rating={self.rating})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """모델을 딕셔너리로 변환"""
        return {
            'id': self.id,
            'recipe_id': self.recipe_id,
            'rating': self.rating,
            'comments': self.comments,
            'emotional_response': self.emotional_response,
            'longevity': self.longevity,
            'sillage': self.sillage,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

def get_db_session() -> SQLSession:
    """데이터베이스 세션 생성 (컨텍스트 매니저 사용 권장)"""
    return SessionLocal()


def init_db() -> None:
    """데이터베이스 초기화 함수"""
    Base.metadata.create_all(engine)


# 하위 호환성을 위한 레거시 인터페이스
Session = SessionLocal  # 기존 코드와의 호환성


if __name__ == "__main__":
    init_db()
    print("데이터베이스가 초기화되었습니다.") 