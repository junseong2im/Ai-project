#!/usr/bin/env python3
"""
환경 설정 관리 모듈
하드코딩된 값들을 환경변수로 외부화
"""

import os
from typing import Optional
from pathlib import Path

class EnvironmentConfig:
    """환경 설정 클래스"""
    
    def __init__(self):
        self.load_env_file()
    
    def load_env_file(self):
        """환경변수 파일 로드 (.env 파일 지원)"""
        env_file = Path(__file__).parent.parent / ".env"
        
        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
            except Exception as e:
                print(f"Warning: .env 파일 로드 실패: {e}")
    
    # 서버 설정
    @property
    def server_host(self) -> str:
        return os.getenv("SERVER_HOST", "127.0.0.1")
    
    @property 
    def server_port(self) -> int:
        return int(os.getenv("SERVER_PORT", "8000"))
    
    @property
    def debug_mode(self) -> bool:
        return os.getenv("DEBUG", "false").lower() == "true"
    
    @property
    def reload_on_change(self) -> bool:
        return os.getenv("RELOAD", "true").lower() == "true"
    
    # 데이터베이스 설정
    @property
    def database_url(self) -> str:
        return os.getenv("DATABASE_URL", "sqlite:///perfume_ai.db")
    
    # 파일 업로드 설정
    @property
    def max_file_size_mb(self) -> int:
        return int(os.getenv("MAX_FILE_SIZE_MB", "500"))
    
    @property
    def upload_temp_dir(self) -> str:
        return os.getenv("UPLOAD_TEMP_DIR", "temp/uploads")
    
    # AI 모델 경로
    @property
    def model_base_path(self) -> str:
        return os.getenv("MODEL_BASE_PATH", "models")
    
    @property
    def data_base_path(self) -> str:
        return os.getenv("DATA_BASE_PATH", "data")
    
    # 로깅 설정
    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def log_file_path(self) -> Optional[str]:
        return os.getenv("LOG_FILE_PATH")  # None이면 콘솔만 사용
    
    # 보안 설정
    @property
    def allow_cors_origins(self) -> list:
        origins = os.getenv("CORS_ORIGINS", "*")
        return origins.split(",") if origins != "*" else ["*"]
    
    @property
    def enable_security_logging(self) -> bool:
        return os.getenv("SECURITY_LOGGING", "true").lower() == "true"
    
    # 성능 설정
    @property
    def enable_caching(self) -> bool:
        return os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    @property
    def cache_max_size(self) -> int:
        return int(os.getenv("CACHE_MAX_SIZE", "1000"))
    
    @property
    def video_analysis_max_frames(self) -> int:
        return int(os.getenv("VIDEO_MAX_FRAMES", "50"))
    
    # OpenAI API 설정 (있는 경우)
    @property
    def openai_api_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def openai_model(self) -> str:
        return os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    def get_model_path(self, model_name: str) -> str:
        """모델 파일 전체 경로 반환"""
        return str(Path(self.model_base_path) / model_name)
    
    def get_data_path(self, data_name: str) -> str:
        """데이터 파일 전체 경로 반환"""
        return str(Path(self.data_base_path) / data_name)
    
    def to_dict(self) -> dict:
        """설정을 딕셔너리로 반환 (디버깅용)"""
        return {
            "server_host": self.server_host,
            "server_port": self.server_port,
            "debug_mode": self.debug_mode,
            "database_url": self.database_url,
            "max_file_size_mb": self.max_file_size_mb,
            "log_level": self.log_level,
            "enable_caching": self.enable_caching,
            "video_max_frames": self.video_analysis_max_frames
        }

# 글로벌 설정 인스턴스
config = EnvironmentConfig()

def get_config() -> EnvironmentConfig:
    """설정 인스턴스 반환"""
    return config