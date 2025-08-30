#!/usr/bin/env python3
"""
파일 업로드 보안 유틸리티
"""

import os
import mimetypes
from typing import Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 지원하는 비디오 포맷과 해당 매직 넘버
VIDEO_SIGNATURES = {
    # MP4
    b'\x00\x00\x00\x18ftypmp4': 'video/mp4',
    b'\x00\x00\x00\x20ftypiso': 'video/mp4',
    # AVI
    b'RIFF': 'video/avi',
    # MOV/QuickTime
    b'\x00\x00\x00\x14ftypqt': 'video/quicktime',
    b'\x00\x00\x00\x20ftypqt': 'video/quicktime',
    # MKV
    b'\x1a\x45\xdf\xa3': 'video/x-matroska',
}

# 안전한 파일 확장자
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.qt'}

# 최대 파일 크기 (바이트)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

def validate_file_signature(file_content: bytes) -> Tuple[bool, Optional[str]]:
    """파일 시그니처(매직 넘버)를 검증하여 실제 파일 타입을 확인"""
    
    if len(file_content) < 32:
        return False, "파일이 너무 작습니다"
    
    # AVI 파일 특별 처리 (RIFF 헤더 + AVI 식별자)
    if file_content.startswith(b'RIFF') and b'AVI ' in file_content[:32]:
        return True, 'video/avi'
    
    # 다른 비디오 포맷들 확인
    for signature, mime_type in VIDEO_SIGNATURES.items():
        if file_content.startswith(signature):
            return True, mime_type
    
    return False, "지원하지 않는 비디오 형식입니다"

def validate_file_extension(filename: str) -> bool:
    """파일 확장자 검증"""
    if not filename:
        return False
    
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_VIDEO_EXTENSIONS

def validate_file_size(file_size: int) -> bool:
    """파일 크기 검증"""
    return 0 < file_size <= MAX_FILE_SIZE

def sanitize_filename(filename: str) -> str:
    """파일명 안전화 처리"""
    if not filename:
        return "unknown_file"
    
    # 위험한 문자들 제거
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    sanitized = ''.join(c if c in safe_chars else '_' for c in filename)
    
    # 길이 제한
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:250] + ext
    
    return sanitized

def comprehensive_video_validation(filename: str, file_content: bytes) -> Tuple[bool, str]:
    """종합적인 비디오 파일 검증"""
    
    # 1. 파일명 검증
    if not validate_file_extension(filename):
        return False, f"지원하지 않는 파일 확장자입니다. 지원 형식: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
    
    # 2. 파일 크기 검증
    if not validate_file_size(len(file_content)):
        return False, f"파일 크기가 너무 큽니다. 최대 {MAX_FILE_SIZE // (1024*1024)}MB까지 지원됩니다."
    
    # 3. 파일 시그니처 검증
    is_valid_signature, signature_result = validate_file_signature(file_content)
    if not is_valid_signature:
        return False, f"파일 내용이 비디오 형식이 아닙니다: {signature_result}"
    
    # 4. MIME 타입 이중 확인
    guessed_type, _ = mimetypes.guess_type(filename)
    if guessed_type and not guessed_type.startswith('video/'):
        return False, f"MIME 타입이 비디오가 아닙니다: {guessed_type}"
    
    logger.info(f"파일 검증 성공: {filename} ({signature_result})")
    return True, f"검증 완료: {signature_result}"

def create_secure_temp_path(original_filename: str) -> Path:
    """안전한 임시 파일 경로 생성"""
    import tempfile
    import uuid
    
    # 파일명 안전화
    safe_filename = sanitize_filename(original_filename)
    
    # UUID를 사용한 고유한 파일명 생성
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(safe_filename)
    secure_filename = f"{name}_{unique_id}{ext}"
    
    # 임시 디렉토리에 안전한 경로 생성
    temp_dir = Path(tempfile.gettempdir()) / "movie_scent_uploads"
    temp_dir.mkdir(exist_ok=True, mode=0o700)  # 소유자만 접근 가능
    
    return temp_dir / secure_filename

# 추가 보안 함수들
def check_suspicious_patterns(file_content: bytes) -> Tuple[bool, str]:
    """의심스러운 패턴 검사 (악성코드 예방)"""
    
    # 스크립트 태그 등 의심스러운 패턴
    suspicious_patterns = [
        b'<script',
        b'javascript:',
        b'<?php',
        b'<%',
        b'#!/bin/',
        b'powershell',
        b'cmd.exe'
    ]
    
    content_sample = file_content[:1024].lower()  # 처음 1KB만 검사
    
    for pattern in suspicious_patterns:
        if pattern in content_sample:
            return False, f"의심스러운 패턴이 발견되었습니다: {pattern.decode('utf-8', errors='ignore')}"
    
    return True, "패턴 검사 통과"

def log_upload_attempt(filename: str, file_size: int, client_ip: str = "unknown", success: bool = True):
    """업로드 시도 로깅 (보안 모니터링)"""
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"FILE_UPLOAD [{status}] - {filename} ({file_size} bytes) from {client_ip}")