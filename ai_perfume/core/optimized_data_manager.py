#!/usr/bin/env python3
"""
효율적인 데이터 구조 관리 시스템
메모리 최적화 + 빠른 검색 + 인덱싱
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import pickle
import hashlib
import time
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import defaultdict
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class SceneData:
    """최적화된 장면 데이터 클래스"""
    scene_id: str
    scene_type: str
    location: str
    time_of_day: str
    weather: str
    emotions: List[str]
    visual_elements: List[str]
    intensity: float
    longevity: float
    projection: float
    primary_notes: List[str]
    secondary_notes: List[str]
    mood: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scene_id': self.scene_id,
            'scene_type': self.scene_type,
            'location': self.location,
            'time_of_day': self.time_of_day,
            'weather': self.weather,
            'emotions': self.emotions,
            'visual_elements': self.visual_elements,
            'intensity': self.intensity,
            'longevity': self.longevity,
            'projection': self.projection,
            'primary_notes': self.primary_notes,
            'secondary_notes': self.secondary_notes,
            'mood': self.mood
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneData':
        return cls(**data)

class OptimizedDataManager:
    """최적화된 데이터 관리 시스템"""
    
    def __init__(self, db_path: str = "data/optimized_perfume.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # 인메모리 캐시
        self.scene_cache: Dict[str, SceneData] = {}
        self.perfume_cache: Dict[str, Dict] = {}
        self.search_index: Dict[str, List[str]] = defaultdict(list)
        
        # 빠른 검색을 위한 인덱스
        self.emotion_index: Dict[str, List[str]] = defaultdict(list)
        self.location_index: Dict[str, List[str]] = defaultdict(list)
        self.note_index: Dict[str, List[str]] = defaultdict(list)
        self.scene_type_index: Dict[str, List[str]] = defaultdict(list)
        
        # 성능 통계
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_queries': 0,
            'avg_query_time': 0.0
        }
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        self._initialize_database()
        self._load_all_data()
        
        logger.info(f"최적화된 데이터 매니저 초기화 완료: {len(self.scene_cache)}개 장면")
    
    def _initialize_database(self):
        """효율적인 데이터베이스 스키마 생성"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # 성능 최적화
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")
            
            # 최적화된 테이블 스키마
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scenes (
                    scene_id TEXT PRIMARY KEY,
                    scene_type TEXT NOT NULL,
                    location TEXT NOT NULL,
                    time_of_day TEXT NOT NULL,
                    weather TEXT NOT NULL,
                    emotions TEXT NOT NULL,  -- JSON 배열
                    visual_elements TEXT NOT NULL,  -- JSON 배열
                    intensity REAL NOT NULL,
                    longevity REAL NOT NULL,
                    projection REAL NOT NULL,
                    primary_notes TEXT NOT NULL,  -- JSON 배열
                    secondary_notes TEXT NOT NULL,  -- JSON 배열
                    mood TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS perfumes (
                    perfume_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    brand TEXT NOT NULL,
                    gender TEXT NOT NULL,
                    rating REAL,
                    rating_count INTEGER,
                    main_accords TEXT,  -- JSON 배열
                    description TEXT,
                    price_range TEXT,
                    availability TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS scene_recommendations (
                    scene_id TEXT,
                    perfume_id TEXT,
                    confidence_score REAL,
                    recommendation_type TEXT,  -- top_pick, alternative, budget, niche
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (scene_id, perfume_id),
                    FOREIGN KEY (scene_id) REFERENCES scenes (scene_id),
                    FOREIGN KEY (perfume_id) REFERENCES perfumes (perfume_id)
                );
                
                -- 성능을 위한 인덱스들
                CREATE INDEX IF NOT EXISTS idx_scenes_type ON scenes (scene_type);
                CREATE INDEX IF NOT EXISTS idx_scenes_location ON scenes (location);
                CREATE INDEX IF NOT EXISTS idx_scenes_mood ON scenes (mood);
                CREATE INDEX IF NOT EXISTS idx_perfumes_brand ON perfumes (brand);
                CREATE INDEX IF NOT EXISTS idx_perfumes_rating ON perfumes (rating DESC);
                CREATE INDEX IF NOT EXISTS idx_recommendations_scene ON scene_recommendations (scene_id);
                CREATE INDEX IF NOT EXISTS idx_recommendations_confidence ON scene_recommendations (confidence_score DESC);
                
                -- 전문 검색을 위한 FTS 테이블
                CREATE VIRTUAL TABLE IF NOT EXISTS scenes_fts USING fts5(
                    scene_id,
                    content,  -- 모든 텍스트 내용 통합
                    tokenize = 'porter'
                );
            """)
    
    def _load_all_data(self):
        """모든 데이터를 메모리로 로드하여 빠른 접근 보장"""
        start_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # 장면 데이터 로드
            scenes = conn.execute("SELECT * FROM scenes").fetchall()
            for scene_row in scenes:
                scene_data = SceneData(
                    scene_id=scene_row['scene_id'],
                    scene_type=scene_row['scene_type'],
                    location=scene_row['location'],
                    time_of_day=scene_row['time_of_day'],
                    weather=scene_row['weather'],
                    emotions=json.loads(scene_row['emotions']),
                    visual_elements=json.loads(scene_row['visual_elements']),
                    intensity=scene_row['intensity'],
                    longevity=scene_row['longevity'],
                    projection=scene_row['projection'],
                    primary_notes=json.loads(scene_row['primary_notes']),
                    secondary_notes=json.loads(scene_row['secondary_notes']),
                    mood=scene_row['mood']
                )
                
                self.scene_cache[scene_data.scene_id] = scene_data
                self._update_indexes(scene_data)
            
            # 향수 데이터 로드
            perfumes = conn.execute("SELECT * FROM perfumes").fetchall()
            for perfume_row in perfumes:
                perfume_data = dict(perfume_row)
                if perfume_data['main_accords']:
                    perfume_data['main_accords'] = json.loads(perfume_data['main_accords'])
                
                self.perfume_cache[perfume_data['perfume_id']] = perfume_data
        
        load_time = time.time() - start_time
        logger.info(f"데이터 로딩 완료: {load_time:.3f}초, "
                   f"{len(self.scene_cache)}개 장면, {len(self.perfume_cache)}개 향수")
    
    def _update_indexes(self, scene_data: SceneData):
        """검색 인덱스 업데이트"""
        scene_id = scene_data.scene_id
        
        # 감정별 인덱스
        for emotion in scene_data.emotions:
            self.emotion_index[emotion.lower()].append(scene_id)
        
        # 위치별 인덱스
        self.location_index[scene_data.location.lower()].append(scene_id)
        
        # 노트별 인덱스
        for note in scene_data.primary_notes + scene_data.secondary_notes:
            self.note_index[note.lower()].append(scene_id)
        
        # 장면 타입별 인덱스
        self.scene_type_index[scene_data.scene_type.lower()].append(scene_id)
    
    def add_scene(self, scene_data: SceneData) -> bool:
        """새로운 장면 추가"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO scenes 
                        (scene_id, scene_type, location, time_of_day, weather, 
                         emotions, visual_elements, intensity, longevity, projection,
                         primary_notes, secondary_notes, mood)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        scene_data.scene_id,
                        scene_data.scene_type,
                        scene_data.location,
                        scene_data.time_of_day,
                        scene_data.weather,
                        json.dumps(scene_data.emotions),
                        json.dumps(scene_data.visual_elements),
                        scene_data.intensity,
                        scene_data.longevity,
                        scene_data.projection,
                        json.dumps(scene_data.primary_notes),
                        json.dumps(scene_data.secondary_notes),
                        scene_data.mood
                    ))
                    
                    # FTS 테이블 업데이트
                    content = f"{scene_data.scene_type} {scene_data.location} {scene_data.mood} " + \
                             " ".join(scene_data.emotions + scene_data.visual_elements + 
                                    scene_data.primary_notes + scene_data.secondary_notes)
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO scenes_fts (scene_id, content)
                        VALUES (?, ?)
                    """, (scene_data.scene_id, content))
                
                # 캐시 및 인덱스 업데이트
                self.scene_cache[scene_data.scene_id] = scene_data
                self._update_indexes(scene_data)
                
                return True
                
        except Exception as e:
            logger.error(f"장면 추가 실패: {e}")
            return False
    
    @lru_cache(maxsize=1000)
    def search_scenes_by_emotion(self, emotion: str) -> List[SceneData]:
        """감정으로 장면 검색 (캐시됨)"""
        start_time = time.time()
        
        with self._lock:
            self.stats['total_queries'] += 1
            
            emotion_lower = emotion.lower()
            if emotion_lower in self.emotion_index:
                self.stats['cache_hits'] += 1
                scene_ids = self.emotion_index[emotion_lower]
                results = [self.scene_cache[sid] for sid in scene_ids if sid in self.scene_cache]
            else:
                self.stats['cache_misses'] += 1
                results = []
            
            query_time = time.time() - start_time
            self._update_avg_query_time(query_time)
            
            return results
    
    @lru_cache(maxsize=1000)
    def search_scenes_by_type(self, scene_type: str) -> List[SceneData]:
        """장면 타입으로 검색 (캐시됨)"""
        start_time = time.time()
        
        with self._lock:
            self.stats['total_queries'] += 1
            
            type_lower = scene_type.lower()
            if type_lower in self.scene_type_index:
                self.stats['cache_hits'] += 1
                scene_ids = self.scene_type_index[type_lower]
                results = [self.scene_cache[sid] for sid in scene_ids if sid in self.scene_cache]
            else:
                self.stats['cache_misses'] += 1
                results = []
            
            query_time = time.time() - start_time
            self._update_avg_query_time(query_time)
            
            return results
    
    def search_scenes_advanced(self, 
                             emotions: Optional[List[str]] = None,
                             scene_types: Optional[List[str]] = None,
                             locations: Optional[List[str]] = None,
                             min_intensity: float = 0,
                             max_intensity: float = 10,
                             limit: int = 50) -> List[SceneData]:
        """고급 복합 검색"""
        start_time = time.time()
        
        with self._lock:
            self.stats['total_queries'] += 1
            
            candidate_scene_ids = set(self.scene_cache.keys())
            
            # 감정 필터
            if emotions:
                emotion_candidates = set()
                for emotion in emotions:
                    emotion_candidates.update(self.emotion_index.get(emotion.lower(), []))
                candidate_scene_ids &= emotion_candidates
            
            # 장면 타입 필터  
            if scene_types:
                type_candidates = set()
                for scene_type in scene_types:
                    type_candidates.update(self.scene_type_index.get(scene_type.lower(), []))
                candidate_scene_ids &= type_candidates
            
            # 위치 필터
            if locations:
                location_candidates = set()
                for location in locations:
                    location_candidates.update(self.location_index.get(location.lower(), []))
                candidate_scene_ids &= location_candidates
            
            # 강도 필터 및 결과 구성
            results = []
            for scene_id in candidate_scene_ids:
                if len(results) >= limit:
                    break
                    
                scene = self.scene_cache.get(scene_id)
                if scene and min_intensity <= scene.intensity <= max_intensity:
                    results.append(scene)
            
            query_time = time.time() - start_time
            self._update_avg_query_time(query_time)
            
            # 강도 순으로 정렬
            results.sort(key=lambda x: x.intensity, reverse=True)
            
            return results[:limit]
    
    def get_scene_by_id(self, scene_id: str) -> Optional[SceneData]:
        """ID로 장면 검색 (O(1) 접근)"""
        with self._lock:
            return self.scene_cache.get(scene_id)
    
    def get_similar_scenes(self, reference_scene: SceneData, limit: int = 10) -> List[Tuple[SceneData, float]]:
        """유사한 장면 검색 (유사도 점수 포함)"""
        start_time = time.time()
        
        similar_scenes = []
        
        with self._lock:
            for scene_id, scene in self.scene_cache.items():
                if scene_id == reference_scene.scene_id:
                    continue
                
                # 유사도 계산
                similarity = self._calculate_similarity(reference_scene, scene)
                if similarity > 0.3:  # 임계값
                    similar_scenes.append((scene, similarity))
            
            # 유사도 순 정렬
            similar_scenes.sort(key=lambda x: x[1], reverse=True)
        
        query_time = time.time() - start_time
        self._update_avg_query_time(query_time)
        
        return similar_scenes[:limit]
    
    def _calculate_similarity(self, scene1: SceneData, scene2: SceneData) -> float:
        """장면 간 유사도 계산"""
        score = 0.0
        
        # 장면 타입 (가중치: 0.3)
        if scene1.scene_type == scene2.scene_type:
            score += 0.3
        
        # 위치 (가중치: 0.2)  
        if scene1.location == scene2.location:
            score += 0.2
        
        # 무드 (가중치: 0.2)
        if scene1.mood == scene2.mood:
            score += 0.2
        
        # 감정 중복 (가중치: 0.15)
        emotion_overlap = len(set(scene1.emotions) & set(scene2.emotions))
        emotion_total = len(set(scene1.emotions) | set(scene2.emotions))
        if emotion_total > 0:
            score += 0.15 * (emotion_overlap / emotion_total)
        
        # 노트 중복 (가중치: 0.15)
        notes1 = set(scene1.primary_notes + scene1.secondary_notes)
        notes2 = set(scene2.primary_notes + scene2.secondary_notes)
        note_overlap = len(notes1 & notes2)
        note_total = len(notes1 | notes2)
        if note_total > 0:
            score += 0.15 * (note_overlap / note_total)
        
        return min(score, 1.0)
    
    def _update_avg_query_time(self, query_time: float):
        """평균 쿼리 시간 업데이트"""
        total_queries = self.stats['total_queries']
        current_avg = self.stats['avg_query_time']
        self.stats['avg_query_time'] = ((current_avg * (total_queries - 1)) + query_time) / total_queries
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        with self._lock:
            cache_hit_rate = (self.stats['cache_hits'] / max(1, self.stats['total_queries'])) * 100
            
            return {
                'total_queries': self.stats['total_queries'],
                'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                'avg_query_time': f"{self.stats['avg_query_time']:.4f}초",
                'scenes_in_memory': len(self.scene_cache),
                'perfumes_in_memory': len(self.perfume_cache),
                'emotion_index_size': len(self.emotion_index),
                'location_index_size': len(self.location_index),
                'note_index_size': len(self.note_index)
            }
    
    def optimize_database(self):
        """데이터베이스 최적화"""
        logger.info("데이터베이스 최적화 시작...")
        
        with sqlite3.connect(self.db_path) as conn:
            # 통계 업데이트
            conn.execute("ANALYZE")
            
            # 인덱스 재구성
            conn.execute("REINDEX")
            
            # 공간 최적화
            conn.execute("VACUUM")
            
        logger.info("데이터베이스 최적화 완료")
    
    def export_data(self, export_path: str):
        """데이터 내보내기"""
        export_data = {
            'scenes': [scene.to_dict() for scene in self.scene_cache.values()],
            'perfumes': list(self.perfume_cache.values()),
            'stats': self.get_performance_stats(),
            'export_timestamp': time.time()
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"데이터 내보내기 완료: {export_path}")

# 글로벌 인스턴스 (싱글톤 패턴)
_data_manager_instance = None

def get_data_manager() -> OptimizedDataManager:
    """글로벌 데이터 매니저 인스턴스 반환"""
    global _data_manager_instance
    if _data_manager_instance is None:
        _data_manager_instance = OptimizedDataManager()
    return _data_manager_instance