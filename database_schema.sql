-- AI 향료 시스템 최적화 데이터베이스 스키마
-- PostgreSQL 15 + pgvector 확장

-- 확장 설치
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- 텍스트 유사도 검색
CREATE EXTENSION IF NOT EXISTS btree_gin; -- 복합 인덱스

-- 1. 향료 마스터 테이블
CREATE TABLE fragrance_materials (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50), -- citrus, floral, woody, etc.
    volatility_level INTEGER CHECK (volatility_level BETWEEN 1 AND 100),
    scent_profile JSONB, -- {"sweet": 0.8, "fresh": 0.6, "warm": 0.3}
    chemical_properties JSONB,
    price_per_ml DECIMAL(10,2),
    supplier_info JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. 영화/장면 메타데이터
CREATE TABLE movie_scenes (
    id SERIAL PRIMARY KEY,
    movie_title VARCHAR(200) NOT NULL,
    scene_description TEXT NOT NULL,
    genre VARCHAR(50) NOT NULL,
    emotions JSONB, -- ["romantic", "intense", "nostalgic"]
    duration_minutes INTEGER,
    complexity_score FLOAT CHECK (complexity_score BETWEEN 0 AND 1),
    keywords TEXT[], -- GIN 인덱스로 빠른 검색
    scene_embedding VECTOR(384), -- OpenAI/Sentence-BERT 임베딩
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3. AI 생성 향료 레시피 (메인 테이블)
CREATE TABLE ai_fragrance_recipes (
    id SERIAL PRIMARY KEY,
    scene_id INTEGER REFERENCES movie_scenes(id),
    recipe_name VARCHAR(200),
    
    -- AI 예측 결과
    confidence_score FLOAT CHECK (confidence_score BETWEEN 0 AND 1),
    prediction_method VARCHAR(50), -- 'deep_learning', 'ensemble', etc.
    model_version VARCHAR(20),
    
    -- 향료 조성 (JSON으로 유연하게)
    fragrance_composition JSONB, -- 전체 레시피 구조
    total_materials INTEGER,
    estimated_duration_minutes INTEGER,
    volatility_profile VARCHAR(20),
    
    -- 비즈니스 데이터
    production_cost DECIMAL(10,2),
    difficulty_level INTEGER CHECK (difficulty_level BETWEEN 1 AND 5),
    manufacturing_notes TEXT,
    
    -- 성능 메트릭
    user_rating DECIMAL(3,2), -- 사용자 평가
    success_rate FLOAT, -- 실제 제조 성공률
    
    -- 메타데이터
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),
    
    -- 검색 최적화
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', 
            COALESCE(recipe_name, '') || ' ' || 
            COALESCE((fragrance_composition->>'description'), '')
        )
    ) STORED
);

-- 4. 향료 조성 상세 (정규화된 테이블)
CREATE TABLE recipe_materials (
    id SERIAL PRIMARY KEY,
    recipe_id INTEGER REFERENCES ai_fragrance_recipes(id) ON DELETE CASCADE,
    material_id INTEGER REFERENCES fragrance_materials(id),
    note_type VARCHAR(20) CHECK (note_type IN ('top', 'middle', 'base')),
    concentration_percent DECIMAL(5,2) CHECK (concentration_percent > 0),
    strength_level VARCHAR(20), -- 'subtle', 'medium', 'strong'
    special_treatment JSONB, -- 특수 처리 방법
    position_order INTEGER, -- 레이어 내 순서
    created_at TIMESTAMP DEFAULT NOW()
);

-- 5. AI 모델 성능 추적
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(20),
    genre VARCHAR(50),
    
    -- 성능 지표
    average_confidence DECIMAL(5,4),
    accuracy_score DECIMAL(5,4),
    prediction_count INTEGER DEFAULT 0,
    success_predictions INTEGER DEFAULT 0,
    
    -- 시간별 성능
    evaluation_date DATE,
    test_dataset_size INTEGER,
    
    -- 성능 상세
    genre_specific_accuracy JSONB, -- {"action": 0.92, "romantic": 0.95}
    confidence_distribution JSONB, -- 신뢰도 분포 히스토그램
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- 6. 사용자 피드백 (학습 개선용)
CREATE TABLE user_feedback (
    id SERIAL PRIMARY KEY,
    recipe_id INTEGER REFERENCES ai_fragrance_recipes(id),
    user_id VARCHAR(100),
    
    -- 피드백 데이터
    accuracy_rating INTEGER CHECK (accuracy_rating BETWEEN 1 AND 5),
    scent_match_rating INTEGER CHECK (scent_match_rating BETWEEN 1 AND 5),
    overall_satisfaction INTEGER CHECK (overall_satisfaction BETWEEN 1 AND 5),
    
    -- 상세 피드백
    feedback_text TEXT,
    suggested_improvements JSONB,
    would_manufacture BOOLEAN,
    
    -- 실제 제조 결과 (있는 경우)
    actually_manufactured BOOLEAN DEFAULT FALSE,
    manufacturing_result JSONB,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- 7. 캐시 테이블 (자주 조회되는 예측 결과)
CREATE TABLE prediction_cache (
    id SERIAL PRIMARY KEY,
    scene_hash VARCHAR(64) UNIQUE, -- MD5 hash of scene description
    cached_prediction JSONB,
    confidence_score DECIMAL(5,4),
    cache_hits INTEGER DEFAULT 0,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ================================
-- 인덱스 생성 (성능 최적화)
-- ================================

-- 1. 벡터 유사도 검색 (가장 중요!)
CREATE INDEX idx_scene_embedding_cosine ON movie_scenes 
USING ivfflat (scene_embedding vector_cosine_ops) WITH (lists = 100);

-- 2. 장르별 빠른 필터링
CREATE INDEX idx_recipes_genre ON ai_fragrance_recipes(scene_id, confidence_score DESC)
WHERE confidence_score >= 0.8;

CREATE INDEX idx_scenes_genre ON movie_scenes(genre, complexity_score);

-- 3. 신뢰도 기반 정렬
CREATE INDEX idx_recipes_confidence ON ai_fragrance_recipes(confidence_score DESC, created_at DESC);

-- 4. 텍스트 검색 최적화
CREATE INDEX idx_recipes_search ON ai_fragrance_recipes USING GIN(search_vector);
CREATE INDEX idx_scene_keywords ON movie_scenes USING GIN(keywords);

-- 5. 조성 검색 (특정 향료 포함 레시피)
CREATE INDEX idx_materials_recipe ON recipe_materials(material_id, note_type, concentration_percent);

-- 6. 성능 모니터링
CREATE INDEX idx_performance_model ON model_performance(model_name, evaluation_date DESC);

-- 7. 캐시 최적화
CREATE INDEX idx_cache_hash ON prediction_cache(scene_hash, expires_at);
CREATE INDEX idx_cache_expiry ON prediction_cache(expires_at) WHERE expires_at > NOW();

-- ================================
-- 뷰 생성 (자주 사용되는 조인)
-- ================================

-- 1. 완전한 레시피 정보 뷰
CREATE VIEW complete_recipes AS
SELECT 
    r.id,
    r.recipe_name,
    r.confidence_score,
    r.prediction_method,
    s.movie_title,
    s.scene_description,
    s.genre,
    s.emotions,
    r.fragrance_composition,
    r.total_materials,
    r.production_cost,
    r.user_rating,
    r.created_at
FROM ai_fragrance_recipes r
JOIN movie_scenes s ON r.scene_id = s.id
WHERE r.confidence_score >= 0.7;  -- 신뢰할만한 레시피만

-- 2. 장르별 성능 요약 뷰
CREATE VIEW genre_performance_summary AS
SELECT 
    s.genre,
    COUNT(*) as total_recipes,
    AVG(r.confidence_score) as avg_confidence,
    AVG(r.user_rating) as avg_user_rating,
    COUNT(CASE WHEN r.confidence_score >= 0.9 THEN 1 END) as high_confidence_count
FROM ai_fragrance_recipes r
JOIN movie_scenes s ON r.scene_id = s.id
GROUP BY s.genre;

-- 3. 인기 향료 조합 뷰
CREATE VIEW popular_material_combinations AS
SELECT 
    m1.name as material1,
    m2.name as material2,
    COUNT(*) as combination_count,
    AVG(r.confidence_score) as avg_confidence
FROM recipe_materials rm1
JOIN recipe_materials rm2 ON rm1.recipe_id = rm2.recipe_id AND rm1.id < rm2.id
JOIN fragrance_materials m1 ON rm1.material_id = m1.id
JOIN fragrance_materials m2 ON rm2.material_id = m2.id
JOIN ai_fragrance_recipes r ON rm1.recipe_id = r.id
GROUP BY m1.name, m2.name
HAVING COUNT(*) >= 10  -- 10번 이상 함께 사용된 조합
ORDER BY combination_count DESC, avg_confidence DESC;

-- ================================
-- 함수 생성 (비즈니스 로직)
-- ================================

-- 1. 유사한 장면 검색 함수
CREATE OR REPLACE FUNCTION find_similar_scenes(
    input_embedding VECTOR(384), 
    limit_count INTEGER DEFAULT 5
)
RETURNS TABLE (
    scene_id INTEGER,
    movie_title VARCHAR(200),
    scene_description TEXT,
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ms.id,
        ms.movie_title,
        ms.scene_description,
        (1 - (ms.scene_embedding <=> input_embedding))::FLOAT as similarity
    FROM movie_scenes ms
    ORDER BY ms.scene_embedding <=> input_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- 2. 레시피 추천 함수 (신뢰도 + 사용자 평가 기반)
CREATE OR REPLACE FUNCTION recommend_recipes(
    target_genre VARCHAR(50),
    min_confidence FLOAT DEFAULT 0.8
)
RETURNS TABLE (
    recipe_id INTEGER,
    recipe_name VARCHAR(200),
    confidence_score FLOAT,
    user_rating DECIMAL(3,2),
    recommendation_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        r.id,
        r.recipe_name,
        r.confidence_score,
        r.user_rating,
        (r.confidence_score * 0.7 + COALESCE(r.user_rating, 3.0) / 5.0 * 0.3) as rec_score
    FROM ai_fragrance_recipes r
    JOIN movie_scenes s ON r.scene_id = s.id
    WHERE s.genre = target_genre 
        AND r.confidence_score >= min_confidence
    ORDER BY rec_score DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- ================================
-- 트리거 (자동 업데이트)
-- ================================

-- 1. 업데이트 시간 자동 갱신
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_recipes_modtime
    BEFORE UPDATE ON ai_fragrance_recipes
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- 2. 캐시 자동 정리 (만료된 항목 삭제)
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM prediction_cache WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- 3. 성능 통계 자동 업데이트
CREATE OR REPLACE FUNCTION update_model_performance()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO model_performance (
        model_name, model_version, genre,
        average_confidence, prediction_count,
        evaluation_date
    )
    SELECT 
        NEW.model_version,
        NEW.model_version,
        s.genre,
        AVG(r.confidence_score),
        COUNT(*),
        CURRENT_DATE
    FROM ai_fragrance_recipes r
    JOIN movie_scenes s ON r.scene_id = s.id
    WHERE r.model_version = NEW.model_version
        AND DATE(r.created_at) = CURRENT_DATE
    GROUP BY s.genre
    ON CONFLICT (model_name, model_version, genre, evaluation_date)
    DO UPDATE SET
        average_confidence = EXCLUDED.average_confidence,
        prediction_count = EXCLUDED.prediction_count,
        updated_at = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ================================
-- 샘플 데이터 삽입
-- ================================

-- 향료 마스터 데이터
INSERT INTO fragrance_materials (name, category, volatility_level, scent_profile) VALUES
('bergamot', 'citrus', 95, '{"fresh": 0.9, "citrusy": 1.0, "bright": 0.8}'),
('rose_absolute', 'floral', 60, '{"romantic": 1.0, "floral": 1.0, "luxurious": 0.9}'),
('cedar', 'woody', 25, '{"woody": 1.0, "warm": 0.7, "grounding": 0.8}'),
('vanilla', 'oriental', 40, '{"sweet": 0.9, "warm": 0.8, "comforting": 1.0}'),
('black_pepper', 'spicy', 85, '{"spicy": 1.0, "energetic": 0.9, "masculine": 0.8}');

-- 영화 장면 샘플
INSERT INTO movie_scenes (movie_title, scene_description, genre, emotions, duration_minutes, complexity_score, keywords) VALUES
('Titanic', '잭과 로즈가 배 앞에서 팔을 벌리며 나누는 운명적 사랑의 순간', 'romantic', '["love", "freedom", "romantic"]', 3, 0.7, ARRAY['love', 'ship', 'ocean', 'romantic']),
('Avengers Endgame', '토니 스타크가 인피니티 스톤으로 타노스를 물리치는 영웅적 희생', 'action', '["heroic", "sacrifice", "intense"]', 2, 0.9, ARRAY['hero', 'sacrifice', 'battle', 'action']);

-- AI 레시피 샘플 (현실적인 데이터)
INSERT INTO ai_fragrance_recipes (
    scene_id, recipe_name, confidence_score, prediction_method, model_version,
    fragrance_composition, total_materials, estimated_duration_minutes, 
    volatility_profile, production_cost, difficulty_level
) VALUES
(1, 'Oceanic Romance', 0.94, 'deep_learning', 'v2.1',
 '{"description": "Fresh oceanic romance with floral heart", "intensity": "medium"}',
 6, 180, 'medium_volatility', 12.50, 3),
(2, 'Heroes Sacrifice', 0.91, 'ensemble_boost', 'v2.1',
 '{"description": "Powerful metallic with warm base", "intensity": "high"}',
 5, 120, 'high_volatility', 15.75, 4);

COMMENT ON DATABASE postgres IS 'AI 향료 예측 시스템 - 105k 레시피 데이터베이스';