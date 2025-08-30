#!/usr/bin/env python3
"""
비디오 영상 분석을 통한 향 생성 시스템
OpenCV와 AI 모델을 활용한 시각적 요소 분석
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import logging
from dataclasses import dataclass
import colorsys
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class FrameAnalysis:
    """단일 프레임 분석 결과"""
    timestamp: float
    dominant_colors: List[Tuple[int, int, int]]
    brightness: float
    contrast: float
    color_temperature: str  # warm, cool, neutral
    mood_indicators: Dict[str, float]
    scene_elements: List[str]
    
@dataclass 
class SceneSegment:
    """영상 장면 세그먼트"""
    start_time: float
    end_time: float
    frame_analyses: List[FrameAnalysis]
    scene_summary: Dict[str, Any]
    recommended_scent: Dict[str, Any]

class VideoScentAnalyzer:
    """비디오 영상에서 향 추천을 생성하는 분석기"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        
        # 색상-향 매핑 데이터베이스
        self.color_scent_map = {
            'red': {'notes': ['rose', 'cherry', 'cinnamon', 'pepper'], 'emotion': 'passion'},
            'blue': {'notes': ['ocean', 'mint', 'eucalyptus', 'clean_musk'], 'emotion': 'calm'},
            'green': {'notes': ['grass', 'pine', 'basil', 'cucumber'], 'emotion': 'fresh'},
            'yellow': {'notes': ['lemon', 'vanilla', 'honey', 'jasmine'], 'emotion': 'joy'},
            'purple': {'notes': ['lavender', 'violet', 'plum', 'incense'], 'emotion': 'mystery'},
            'orange': {'notes': ['orange', 'peach', 'amber', 'woody'], 'emotion': 'energy'},
            'brown': {'notes': ['coffee', 'chocolate', 'tobacco', 'leather'], 'emotion': 'comfort'},
            'black': {'notes': ['oud', 'patchouli', 'dark_chocolate', 'smoke'], 'emotion': 'intensity'},
            'white': {'notes': ['lily', 'cotton', 'clean_soap', 'aldehydes'], 'emotion': 'purity'}
        }
        
        # 밝기-향 강도 매핑
        self.brightness_intensity_map = {
            'very_dark': 8,    # 진한 향
            'dark': 7,
            'medium': 6,
            'bright': 4,       # 가벼운 향
            'very_bright': 3
        }

    def analyze_video_file(self, video_path: str, max_frames: int = 100) -> List[FrameAnalysis]:
        """비디오 파일 분석"""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
            
        if Path(video_path).suffix.lower() not in self.supported_formats:
            raise ValueError(f"지원하지 않는 비디오 형식입니다: {Path(video_path).suffix}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("비디오 파일을 열 수 없습니다")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # 프레임 샘플링 간격 계산
            frame_interval = max(1, total_frames // max_frames)
            
            analyses = []
            frame_count = 0
            
            logger.info(f"비디오 분석 시작: {duration:.1f}초, {total_frames} 프레임")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 샘플링된 프레임만 분석
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    analysis = self._analyze_frame(frame, timestamp)
                    analyses.append(analysis)
                    
                    if len(analyses) >= max_frames:
                        break
                
                frame_count += 1
            
            logger.info(f"비디오 분석 완료: {len(analyses)}개 프레임 분석")
            return analyses
            
        finally:
            cap.release()

    def _analyze_frame(self, frame: np.ndarray, timestamp: float) -> FrameAnalysis:
        """단일 프레임 분석"""
        # 색상 분석
        dominant_colors = self._extract_dominant_colors(frame, k=5)
        
        # 밝기/대비 계산
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0
        
        # 색온도 판단
        color_temp = self._analyze_color_temperature(frame)
        
        # 무드 인디케이터 계산
        mood_indicators = self._calculate_mood_indicators(dominant_colors, brightness, contrast)
        
        # 장면 요소 추출 (색상 기반)
        scene_elements = self._extract_scene_elements(dominant_colors, brightness)
        
        return FrameAnalysis(
            timestamp=timestamp,
            dominant_colors=dominant_colors,
            brightness=brightness,
            contrast=contrast,
            color_temperature=color_temp,
            mood_indicators=mood_indicators,
            scene_elements=scene_elements
        )

    def _extract_dominant_colors(self, frame: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """K-means를 사용한 주요 색상 추출"""
        # 이미지 크기 조정 (성능 향상)
        small_frame = cv2.resize(frame, (150, 150))
        data = small_frame.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means 클러스터링
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # BGR을 RGB로 변환
        centers = np.uint8(centers)
        colors = []
        for center in centers:
            # BGR to RGB
            colors.append((int(center[2]), int(center[1]), int(center[0])))
        
        return colors

    def _analyze_color_temperature(self, frame: np.ndarray) -> str:
        """색온도 분석 (따뜻함/차가움)"""
        # 평균 RGB 값 계산
        mean_b, mean_g, mean_r = cv2.mean(frame)[:3]
        
        # 적색-청색 비율로 색온도 판단
        if mean_r > mean_b * 1.2:
            return "warm"
        elif mean_b > mean_r * 1.2:
            return "cool"
        else:
            return "neutral"

    def _calculate_mood_indicators(self, colors: List[Tuple[int, int, int]], 
                                 brightness: float, contrast: float) -> Dict[str, float]:
        """색상과 밝기를 기반으로 무드 점수 계산"""
        mood_scores = {
            'romantic': 0.0,
            'energetic': 0.0,
            'calm': 0.0,
            'mysterious': 0.0,
            'fresh': 0.0,
            'dramatic': 0.0
        }
        
        # 색상 기반 무드 점수
        for r, g, b in colors:
            # HSV 변환으로 색조 분석
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hue = h * 360
            
            # 색조별 무드 가중치
            if 330 <= hue or hue <= 30:  # 적색
                mood_scores['romantic'] += 0.3
                mood_scores['energetic'] += 0.2
            elif 30 < hue <= 90:  # 황색-녹색
                mood_scores['energetic'] += 0.3 if s > 0.5 else 0.1
                mood_scores['fresh'] += 0.3
            elif 90 < hue <= 150:  # 녹색
                mood_scores['calm'] += 0.3
                mood_scores['fresh'] += 0.4
            elif 150 < hue <= 270:  # 청색-보라색
                mood_scores['calm'] += 0.4
                mood_scores['mysterious'] += 0.2
            elif 270 < hue <= 330:  # 보라색
                mood_scores['mysterious'] += 0.4
                mood_scores['romantic'] += 0.2
        
        # 밝기 기반 보정
        if brightness < 0.3:  # 어두운 장면
            mood_scores['mysterious'] += 0.3
            mood_scores['dramatic'] += 0.4
        elif brightness > 0.7:  # 밝은 장면
            mood_scores['fresh'] += 0.2
            mood_scores['energetic'] += 0.2
        
        # 대비 기반 보정
        if contrast > 0.3:  # 높은 대비
            mood_scores['dramatic'] += 0.3
            mood_scores['energetic'] += 0.2
        
        # 정규화 (0-1 범위)
        max_score = max(mood_scores.values()) if max(mood_scores.values()) > 0 else 1
        for mood in mood_scores:
            mood_scores[mood] = min(1.0, mood_scores[mood] / max_score)
        
        return mood_scores

    def _extract_scene_elements(self, colors: List[Tuple[int, int, int]], brightness: float) -> List[str]:
        """색상과 밝기를 기반으로 장면 요소 추출"""
        elements = []
        
        # 색상 기반 장면 요소 추정
        color_names = []
        for r, g, b in colors:
            # 가장 가까운 기본 색상 찾기
            distances = {}
            for color_name in ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'black', 'white']:
                if color_name == 'red':
                    ref_color = (255, 0, 0)
                elif color_name == 'blue':
                    ref_color = (0, 0, 255)
                elif color_name == 'green':
                    ref_color = (0, 255, 0)
                elif color_name == 'yellow':
                    ref_color = (255, 255, 0)
                elif color_name == 'purple':
                    ref_color = (128, 0, 128)
                elif color_name == 'orange':
                    ref_color = (255, 165, 0)
                elif color_name == 'brown':
                    ref_color = (139, 69, 19)
                elif color_name == 'black':
                    ref_color = (0, 0, 0)
                elif color_name == 'white':
                    ref_color = (255, 255, 255)
                
                distance = np.sqrt((r - ref_color[0])**2 + (g - ref_color[1])**2 + (b - ref_color[2])**2)
                distances[color_name] = distance
            
            closest_color = min(distances, key=distances.get)
            color_names.append(closest_color)
        
        # 가장 빈번한 색상들을 기반으로 장면 추정
        color_counter = Counter(color_names)
        dominant_color = color_counter.most_common(1)[0][0]
        
        # 색상별 장면 요소 매핑
        scene_mappings = {
            'blue': ['sky', 'ocean', 'water', 'outdoor'],
            'green': ['nature', 'forest', 'outdoor', 'grass'],
            'brown': ['indoor', 'wood', 'earth', 'vintage'],
            'white': ['clean', 'modern', 'minimal', 'bright'],
            'black': ['night', 'indoor', 'dramatic', 'urban'],
            'red': ['passion', 'fire', 'indoor', 'dramatic'],
            'yellow': ['sunny', 'bright', 'outdoor', 'cheerful'],
            'purple': ['night', 'luxury', 'dramatic', 'artistic'],
            'orange': ['sunset', 'warm', 'cozy', 'autumn']
        }
        
        if dominant_color in scene_mappings:
            elements.extend(scene_mappings[dominant_color])
        
        # 밝기 기반 추가 요소
        if brightness < 0.3:
            elements.extend(['night', 'indoor', 'dramatic'])
        elif brightness > 0.7:
            elements.extend(['bright', 'outdoor', 'daylight'])
        
        return list(set(elements))  # 중복 제거

    def generate_scent_from_frames(self, frame_analyses: List[FrameAnalysis]) -> Dict[str, Any]:
        """프레임 분석 결과를 기반으로 향 조합 생성"""
        if not frame_analyses:
            return {"error": "분석할 프레임이 없습니다"}
        
        # 전체 색상 수집 및 통계
        all_colors = []
        all_moods = {'romantic': 0, 'energetic': 0, 'calm': 0, 'mysterious': 0, 'fresh': 0, 'dramatic': 0}
        all_elements = []
        brightness_values = []
        
        for analysis in frame_analyses:
            all_colors.extend(analysis.dominant_colors)
            all_elements.extend(analysis.scene_elements)
            brightness_values.append(analysis.brightness)
            
            for mood, score in analysis.mood_indicators.items():
                all_moods[mood] += score
        
        # 평균 무드 점수 계산
        frame_count = len(frame_analyses)
        avg_moods = {mood: score/frame_count for mood, score in all_moods.items()}
        
        # 주요 무드 선택
        primary_mood = max(avg_moods, key=avg_moods.get)
        
        # 평균 밝기
        avg_brightness = np.mean(brightness_values)
        
        # 색상 기반 향료 선택
        scent_notes = []
        color_counter = Counter()
        
        for r, g, b in all_colors:
            # RGB를 색상명으로 변환 (단순화된 방식)
            color_name = self._rgb_to_color_name(r, g, b)
            color_counter[color_name] += 1
        
        # 상위 3개 색상을 기반으로 향료 선택
        top_colors = color_counter.most_common(3)
        
        for color_name, count in top_colors:
            if color_name in self.color_scent_map:
                color_notes = self.color_scent_map[color_name]['notes']
                # 빈도에 따라 노트 수 결정
                num_notes = min(3, int(count/5) + 1)
                scent_notes.extend(color_notes[:num_notes])
        
        # 무드에 따른 추가 노트
        mood_notes = {
            'romantic': ['rose', 'jasmine', 'vanilla', 'musk'],
            'energetic': ['citrus', 'ginger', 'pepper', 'mint'],
            'calm': ['lavender', 'sandalwood', 'chamomile', 'cedar'],
            'mysterious': ['oud', 'incense', 'patchouli', 'dark_berries'],
            'fresh': ['cucumber', 'green_tea', 'eucalyptus', 'marine'],
            'dramatic': ['leather', 'smoke', 'dark_chocolate', 'tobacco']
        }
        
        if primary_mood in mood_notes:
            scent_notes.extend(mood_notes[primary_mood][:2])
        
        # 중복 제거 및 최종 조합
        unique_notes = list(dict.fromkeys(scent_notes))[:8]  # 상위 8개
        
        # 밝기에 따른 강도 조절
        if avg_brightness < 0.3:
            intensity_level = "strong"
            concentration = 8
        elif avg_brightness < 0.6:
            intensity_level = "medium"
            concentration = 6
        else:
            intensity_level = "light"
            concentration = 4
        
        # 최종 향수 포뮬러 생성
        formula = {
            "scene_analysis": {
                "primary_mood": primary_mood,
                "mood_scores": avg_moods,
                "dominant_colors": [color for color, count in top_colors],
                "scene_elements": list(set(all_elements)),
                "brightness_level": self._get_brightness_level(avg_brightness),
                "analyzed_frames": frame_count
            },
            "scent_profile": {
                "primary_notes": unique_notes[:4],
                "secondary_notes": unique_notes[4:8] if len(unique_notes) > 4 else [],
                "intensity": concentration,
                "projection": min(8, concentration + 1),
                "longevity": max(4, concentration - 1),
                "mood": primary_mood
            },
            "composition": {
                "top_notes": [{"name": note, "percentage": 20} for note in unique_notes[:2]],
                "middle_notes": [{"name": note, "percentage": 35} for note in unique_notes[2:5]],
                "base_notes": [{"name": note, "percentage": 45} for note in unique_notes[5:8]]
            },
            "confidence": min(0.95, 0.6 + (frame_count / 100) * 0.3)  # 프레임 수가 많을수록 신뢰도 상승
        }
        
        return formula

    def _rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """RGB 값을 기본 색상명으로 변환"""
        # 단순화된 색상 분류
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        if max_val - min_val < 30:  # 채도가 낮은 경우
            if max_val < 80:
                return 'black'
            elif max_val > 200:
                return 'white'
            else:
                return 'gray'
        
        # 주요 색상 판별
        if r > g and r > b:
            if g > 100:  # 노란색 성분이 있는 경우
                return 'orange'
            else:
                return 'red'
        elif g > r and g > b:
            if r > 100:  # 노란색 성분
                return 'yellow'
            elif b > 100:  # 청록색 성분
                return 'green'
            else:
                return 'green'
        elif b > r and b > g:
            if r > g:  # 보라색 성분
                return 'purple'
            else:
                return 'blue'
        else:
            return 'brown'

    def _get_brightness_level(self, brightness: float) -> str:
        """밝기 수치를 레벨로 변환"""
        if brightness < 0.2:
            return "very_dark"
        elif brightness < 0.4:
            return "dark"
        elif brightness < 0.7:
            return "medium"
        elif brightness < 0.9:
            return "bright"
        else:
            return "very_bright"

    def create_scene_segments(self, frame_analyses: List[FrameAnalysis], 
                            segment_length: float = 10.0) -> List[SceneSegment]:
        """프레임 분석을 장면 세그먼트로 그룹화"""
        if not frame_analyses:
            return []
        
        segments = []
        current_segment_frames = []
        segment_start = 0.0
        
        for analysis in frame_analyses:
            if not current_segment_frames or analysis.timestamp - segment_start < segment_length:
                current_segment_frames.append(analysis)
            else:
                # 현재 세그먼트 완료
                if current_segment_frames:
                    scent = self.generate_scent_from_frames(current_segment_frames)
                    segment = SceneSegment(
                        start_time=segment_start,
                        end_time=current_segment_frames[-1].timestamp,
                        frame_analyses=current_segment_frames.copy(),
                        scene_summary=self._summarize_segment(current_segment_frames),
                        recommended_scent=scent
                    )
                    segments.append(segment)
                
                # 새 세그먼트 시작
                current_segment_frames = [analysis]
                segment_start = analysis.timestamp
        
        # 마지막 세그먼트 처리
        if current_segment_frames:
            scent = self.generate_scent_from_frames(current_segment_frames)
            segment = SceneSegment(
                start_time=segment_start,
                end_time=current_segment_frames[-1].timestamp,
                frame_analyses=current_segment_frames,
                scene_summary=self._summarize_segment(current_segment_frames),
                recommended_scent=scent
            )
            segments.append(segment)
        
        return segments

    def _summarize_segment(self, frame_analyses: List[FrameAnalysis]) -> Dict[str, Any]:
        """세그먼트 요약 정보 생성"""
        all_elements = []
        all_moods = {'romantic': 0, 'energetic': 0, 'calm': 0, 'mysterious': 0, 'fresh': 0, 'dramatic': 0}
        brightness_values = []
        
        for analysis in frame_analyses:
            all_elements.extend(analysis.scene_elements)
            brightness_values.append(analysis.brightness)
            for mood, score in analysis.mood_indicators.items():
                all_moods[mood] += score
        
        frame_count = len(frame_analyses)
        avg_moods = {mood: score/frame_count for mood, score in all_moods.items()}
        primary_mood = max(avg_moods, key=avg_moods.get)
        
        element_counter = Counter(all_elements)
        top_elements = [elem for elem, count in element_counter.most_common(5)]
        
        return {
            "duration": frame_analyses[-1].timestamp - frame_analyses[0].timestamp,
            "frame_count": frame_count,
            "primary_mood": primary_mood,
            "mood_scores": avg_moods,
            "dominant_elements": top_elements,
            "average_brightness": np.mean(brightness_values),
            "brightness_stability": 1.0 - np.std(brightness_values)  # 밝기 안정성
        }

# 편의 함수들
def analyze_video_for_scent(video_path: str, max_frames: int = 50) -> Dict[str, Any]:
    """비디오 파일을 분석하고 향 추천을 반환하는 편의 함수"""
    analyzer = VideoScentAnalyzer()
    
    try:
        # 비디오 분석
        frame_analyses = analyzer.analyze_video_file(video_path, max_frames)
        
        # 향 생성
        scent_recommendation = analyzer.generate_scent_from_frames(frame_analyses)
        
        # 장면 세그먼트 생성
        segments = analyzer.create_scene_segments(frame_analyses, segment_length=15.0)
        
        return {
            "success": True,
            "video_path": video_path,
            "total_frames_analyzed": len(frame_analyses),
            "overall_scent": scent_recommendation,
            "scene_segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "summary": seg.scene_summary,
                    "scent": seg.recommended_scent
                }
                for seg in segments
            ],
            "analysis_method": "computer_vision_based"
        }
        
    except Exception as e:
        logger.error(f"비디오 향 분석 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "video_path": video_path
        }

def get_video_analyzer() -> VideoScentAnalyzer:
    """VideoScentAnalyzer 인스턴스 반환"""
    return VideoScentAnalyzer()