from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ModelConfig:
    """모델 관련 설정"""
    name: str = "gogamza/kobart-base-v2"
    max_length: int = 512
    batch_size: int = 32
    learning_rate: float = 1e-4


@dataclass(frozen=True)
class DatabaseConfig:
    """데이터베이스 관련 설정"""
    url: str = "sqlite:///perfume_ai.db"
    echo: bool = False


@dataclass(frozen=True)
class NotesConfig:
    """향수 노트 카테고리 설정"""
    categories: Dict[str, List[str]] = None
    
    def __post_init__(self) -> None:
        if self.categories is None:
            object.__setattr__(self, 'categories', {
                'top': ['citrus', 'fresh', 'spicy', 'green', 'marine'],
                'middle': ['floral', 'fruity', 'spicy', 'woody', 'herbal'],
                'base': ['woody', 'sweet', 'musky', 'balsamic', 'earthy']
            })


@dataclass(frozen=True)
class LearningConfig:
    """학습 관련 설정"""
    feedback_weight: float = 0.2
    learning_rate_decay: float = 0.95
    min_feedback_count: int = 10


@dataclass(frozen=True)
class Config:
    """전체 설정"""
    model: ModelConfig = ModelConfig()
    database: DatabaseConfig = DatabaseConfig()
    notes: NotesConfig = NotesConfig()
    learning: LearningConfig = LearningConfig()
    
    # 감정 카테고리
    emotions: List[str] = None
    
    def __post_init__(self) -> None:
        if self.emotions is None:
            object.__setattr__(self, 'emotions', [
                'joy', 'peaceful', 'energy', 'melancholy', 
                'mystery', 'romance', 'nature', 'urban'
            ])
    
    @property
    def data_path(self) -> Path:
        """데이터 디렉토리 경로"""
        return Path("data")
    
    @property
    def model_save_path(self) -> Path:
        """모델 저장 경로"""
        return Path("models") / "saved"


# 하위 호환성을 위한 기존 클래스 형태 유지
class LegacyConfig:
    """기존 코드와의 호환성을 위한 레거시 설정"""
    _config = Config()
    
    # 모델 설정
    MODEL_NAME = _config.model.name
    MAX_LENGTH = _config.model.max_length
    BATCH_SIZE = _config.model.batch_size
    LEARNING_RATE = _config.model.learning_rate
    
    # 데이터베이스 설정
    DATABASE_URL = _config.database.url
    
    # 향수 노트 카테고리
    NOTE_CATEGORIES = _config.notes.categories
    
    # 감정 카테고리
    EMOTIONS = _config.emotions
    
    # 학습 파라미터
    FEEDBACK_WEIGHT = _config.learning.feedback_weight
    LEARNING_RATE_DECAY = _config.learning.learning_rate_decay
    MIN_FEEDBACK_COUNT = _config.learning.min_feedback_count


# 기본 인스턴스 생성
config = Config()

# 하위 호환성을 위해 기존 Config 클래스 유지
Config = LegacyConfig 