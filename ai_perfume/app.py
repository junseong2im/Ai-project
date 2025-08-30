#!/usr/bin/env python3
"""
영화 향수 AI 웹 애플리케이션
FastAPI + React-like interface
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import json
import time
import logging
import asyncio
from pathlib import Path

# 환경 설정 로드
from config.environment import get_config
config = get_config()

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        *([logging.FileHandler(config.log_file_path)] if config.log_file_path else [])
    ]
)
logger = logging.getLogger(__name__)

# 우리 시스템 임포트
try:
    from core.optimized_data_manager import get_data_manager, SceneData
    from core.real_time_movie_scent import RealTimeMovieScentRecommender
    from core.deep_learning_integration import DeepLearningPerfumePredictor, get_trained_predictor
    from core.scent_simulator import ScentSimulator
    from core.standalone_scent_simulator import StandaloneScentSimulator
    from core.video_scent_analyzer import VideoScentAnalyzer, analyze_video_for_scent
    from core.deep_learning_integration import get_multimodal_predictor
except ImportError as e:
    logger.warning(f"Some modules could not be imported: {e}")
    # Vercel에서는 독립형 시뮬레이터만 사용
    try:
        from core.standalone_scent_simulator import StandaloneScentSimulator
    except ImportError:
        StandaloneScentSimulator = None
    
    get_data_manager = None
    SceneData = None
    RealTimeMovieScentRecommender = None
    DeepLearningPerfumePredictor = None
    ScentSimulator = None

# FastAPI 앱 생성
app = FastAPI(
    title="영화 향수 AI API",
    description="감독이 원하는 어떤 향이든 구현해주는 전문 AI 시스템",
    version="2.0.0"
)

# CORS 설정 (환경변수 기반)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allow_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 글로벌 시스템 인스턴스
data_manager = None
recommender = None
dl_predictor = None
trained_predictor = None
scent_simulator = None
video_analyzer = None
multimodal_predictor = None

# 요청/응답 모델들
class SceneRequest(BaseModel):
    description: str
    scene_type: Optional[str] = "drama"
    emotions: Optional[List[str]] = []
    location: Optional[str] = ""
    time_of_day: Optional[str] = ""
    weather: Optional[str] = ""
    intensity_preference: Optional[int] = 5

class PerfumeRecommendation(BaseModel):
    brand: str
    name: str
    category: str
    confidence: float
    price_range: Optional[str] = "mid"

class SceneResponse(BaseModel):
    scene_analysis: Dict[str, Any]
    scent_profile: Dict[str, Any]
    recommendations: Dict[str, List[PerfumeRecommendation]]
    processing_time: float
    confidence: float

class HealthResponse(BaseModel):
    status: str
    systems: Dict[str, bool]
    performance: Dict[str, Any]

# Vercel을 위한 앱 인스턴스
app_instance = app

@app.on_event("startup")
async def startup_event():
    """앱 시작시 시스템 초기화"""
    global data_manager, recommender, dl_predictor, trained_predictor, scent_simulator, video_analyzer, multimodal_predictor
    
    logger.info("Movie Scent AI System Starting...")
    
    try:
        # 1. 향 시뮬레이터 초기화 (의존성 없는 독립형 우선)
        if StandaloneScentSimulator:
            scent_simulator = StandaloneScentSimulator()
            logger.info("Standalone scent simulator initialized")
        elif ScentSimulator:
            scent_simulator = ScentSimulator()
            logger.info("Regular scent simulator initialized")
        else:
            scent_simulator = None
            logger.warning("No scent simulator available")
        
        # 1-1. 비디오 분석기 초기화
        try:
            video_analyzer = VideoScentAnalyzer()
            logger.info("✅ Video scent analyzer initialized")
        except Exception as video_error:
            logger.warning(f"⚠️ Video analyzer initialization failed: {video_error}")
            video_analyzer = None
            
        # 1-2. 멀티모달 예측기 초기화
        try:
            multimodal_predictor = get_multimodal_predictor()
            logger.info("🚀 Multimodal predictor initialized")
        except Exception as multimodal_error:
            logger.warning(f"⚠️ Multimodal predictor initialization failed: {multimodal_error}")
            multimodal_predictor = None
        
        # 2. 전체 AI 시스템 초기화 시도 (로컬 환경에서만)
        try:
            if get_data_manager:
                data_manager = get_data_manager()
                logger.info("✅ 데이터 매니저 초기화 완료")
                
            if RealTimeMovieScentRecommender:
                recommender = RealTimeMovieScentRecommender()
                model_loaded = recommender.load_model_and_preprocessor()
                logger.info(f"✅ 실시간 추천 시스템: {'모델 로드됨' if model_loaded else '기본 모드'}")
                
            if DeepLearningPerfumePredictor:
                try:
                    dl_predictor = DeepLearningPerfumePredictor(
                        "models/perfume_dl_model.pth",
                        "data/processed/preprocessor_tools.pkl",
                        "data/processed/metadata.json"
                    )
                    logger.info("✅ 딥러닝 예측기 초기화 완료")
                except Exception as dl_error:
                    logger.warning(f"⚠️ 딥러닝 예측기 초기화 실패: {dl_error}")
                    dl_predictor = None
            
            # 3. 훈련된 모델 초기화 시도 (200k 데이터셋)
            try:
                if get_trained_predictor:
                    trained_predictor = get_trained_predictor()
                    if trained_predictor.is_loaded:
                        logger.info("🚀 **200k 훈련 모델 로드 완료!**")
                    else:
                        logger.info("⏳ 훈련 모델 로드 대기 중 (훈련 진행중일 수 있음)")
            except Exception as trained_error:
                logger.warning(f"⚠️ 훈련된 모델 초기화 실패: {trained_error}")
                trained_predictor = None
                    
            # 기본 데이터 로드
            if data_manager:
                await load_default_scenes()
                
            if data_manager and recommender and (dl_predictor or trained_predictor):
                system_type = "FULL_AI_WITH_TRAINED_MODEL" if trained_predictor and trained_predictor.is_loaded else "FULL_AI"
                logger.info(f"🎉 **전체 AI 시스템 초기화 완료!** ({system_type})")
            else:
                logger.info("⚠️ 부분 시스템만 활성화됨 (Vercel 모드)")
                
        except Exception as ai_error:
            logger.warning(f"AI 시스템 초기화 실패 (독립 모드로 전환): {ai_error}")
            data_manager = None
            recommender = None
            dl_predictor = None
        
        logger.info("System initialization completed")
        
    except Exception as e:
        logger.error(f"Critical initialization failure: {e}")
        # 최소 기능이라도 유지

async def load_default_scenes():
    """기본 영화 장면 데이터 로드"""
    try:
        movie_data_path = Path("data/movie_scent_database.json")
        if movie_data_path.exists():
            with open(movie_data_path, 'r', encoding='utf-8') as f:
                movie_data = json.load(f)
            
            # 장면 데이터를 데이터 매니저에 추가
            for scene in movie_data.get('movie_scenes', []):
                scene_data = SceneData(
                    scene_id=str(scene['scene_id']),
                    scene_type=scene['scene_type'],
                    location=scene['location'],
                    time_of_day=scene['time_of_day'],
                    weather=scene['weather'],
                    emotions=scene['emotions'],
                    visual_elements=scene['visual_elements'],
                    intensity=scene['scent_profile']['intensity'],
                    longevity=scene['scent_profile']['longevity'],
                    projection=scene['scent_profile']['projection'],
                    primary_notes=scene['scent_profile']['primary_notes'],
                    secondary_notes=scene['scent_profile']['secondary_notes'],
                    mood=scene['scent_profile']['mood']
                )
                
                data_manager.add_scene(scene_data)
            
            logger.info(f"✅ 기본 장면 데이터 로드 완료: {len(movie_data.get('movie_scenes', []))}개")
        
    except Exception as e:
        logger.error(f"기본 데이터 로드 실패: {e}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """메인 웹 인터페이스"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """시스템 상태 확인"""
    systems_status = {
        "data_manager": data_manager is not None,
        "recommender": recommender is not None,
        "deep_learning": dl_predictor is not None
    }
    
    performance = {}
    if data_manager:
        performance = data_manager.get_performance_stats()
    
    return HealthResponse(
        status="healthy" if all(systems_status.values()) else "partial",
        systems=systems_status,
        performance=performance
    )

@app.post("/recommend_scent")
async def recommend_scent(request: dict):
    """향수 추천 API - 실제 시뮬레이션 기반"""
    start_time = time.time()
    
    try:
        description = request.get("description", "")
        scene_type = request.get("scene_type", "drama")
        intensity = request.get("intensity", "medium")
        
        if not description.strip():
            return {"error": "장면 설명을 입력해주세요", "recommendations": []}
        
        # 전체 AI 시스템이 활성화된 경우 (고급 기능 사용)
        if recommender and data_manager:
            logger.info("🚀 Using FULL AI System (Advanced Mode)")
            
            # 실제 딥러닝 추천 시스템 사용
            try:
                result = recommender.recommend_for_scene(
                    description=description,
                    scene_type=scene_type,
                    mood="neutral",
                    intensity_preference=request.get("intensity_preference", 5)
                )
                
                processing_time = time.time() - start_time
                
                # 딥러닝 예측기 추가 분석
                if dl_predictor:
                    try:
                        enhanced_result = dl_predictor.enhanced_predict(description, scene_type)
                        result["enhanced_analysis"] = enhanced_result
                        logger.info("✅ 딥러닝 예측기 결과 추가됨")
                    except Exception as dl_error:
                        logger.warning(f"딥러닝 예측 실패: {dl_error}")
                
                # 200k 훈련된 모델 추가 분석
                if trained_predictor and trained_predictor.is_loaded:
                    try:
                        trained_result = trained_predictor.predict_scene_fragrance(description)
                        if trained_result["success"]:
                            result["trained_model_predictions"] = trained_result["predictions"]
                            result["model_enhanced"] = True
                            logger.info("🚀 **200k 훈련 모델 예측 완료!**")
                    except Exception as trained_error:
                        logger.warning(f"훈련된 모델 예측 실패: {trained_error}")
                
                # 향료 원료 기반 결과 변환 (브랜드 제품 대신 원료 조합)
                scent_profile = result.get("scent_profile", {})
                
                # 실제 향료 원료들 추출
                raw_materials = []
                if "primary_notes" in scent_profile:
                    for note in scent_profile["primary_notes"][:8]:
                        raw_materials.append({
                            "name": note,
                            "category": "primary",
                            "concentration": scent_profile.get("intensity", 7) * 10,
                            "volatility": "top" if note in ["bergamot", "lemon", "lime"] else "middle",
                            "extraction_method": "steam_distillation",
                            "origin": "natural"
                        })
                
                if "secondary_notes" in scent_profile:
                    for note in scent_profile["secondary_notes"][:6]:
                        raw_materials.append({
                            "name": note,
                            "category": "secondary", 
                            "concentration": scent_profile.get("intensity", 7) * 6,
                            "volatility": "base",
                            "extraction_method": "solvent_extraction",
                            "origin": "synthetic"
                        })
                
                # 조합 공식 생성
                mixing_formula = _generate_raw_material_formula(raw_materials, scent_profile)
                
                return {
                    "raw_materials": raw_materials,
                    "mixing_formula": mixing_formula,
                    "scent_profile": {
                        "intensity": scent_profile.get("intensity", 7),
                        "longevity": scent_profile.get("longevity", 6),
                        "projection": scent_profile.get("projection", 5),
                        "harmony": scent_profile.get("confidence", 0.8)
                    },
                    "scene_analysis": result["scene_analysis"],
                    "processing_time": processing_time,
                    "system_mode": "FULL_AI_RAW_MATERIALS"
                }
                
            except Exception as ai_error:
                logger.error(f"전체 AI 시스템 오류, 독립 모드로 전환: {ai_error}")
                # 폴백하여 독립형 시뮬레이터 사용
        
        # 향 시뮬레이터를 사용한 실제 조합 생성 (독립 모드)
        if scent_simulator:
            logger.info("🔬 Using Scent Simulator (Standalone Mode)")
            
            # 감정 추출 (확장된 키워드 기반)
            emotions = []
            desc_lower = description.lower()
            
            emotion_keywords = {
                "love": ["사랑", "로맨틱", "키스", "포옹", "데이트", "만남"],
                "sad": ["슬픈", "이별", "눈물", "우울", "외로운", "그리움"],
                "fear": ["무서운", "공포", "어둠", "귀신", "죽음", "피"],
                "anger": ["화난", "분노", "싸움", "복수", "증오"],
                "joy": ["행복", "웃음", "기쁜", "축하", "파티", "즐거운"],
                "calm": ["평화", "고요", "차분", "명상", "휴식"],
                "tension": ["긴장", "추격", "서스펜스", "스릴"]
            }
            
            for emotion, keywords in emotion_keywords.items():
                if any(word in desc_lower for word in keywords):
                    emotions.append(emotion)
            
            # 독립형 시뮬레이터 사용 (고급 기능)
            if hasattr(scent_simulator, 'run_advanced_simulation'):
                simulation_result = scent_simulator.run_advanced_simulation(
                    description, scene_type, emotions, iterations=150
                )
                
                if "error" not in simulation_result:
                    composition = simulation_result["composition"]
                    
                    # 사용자 친화적 형태로 변환
                    recommendations = []
                    
                    # 모든 노트를 개별 추천으로 표시
                    all_notes = (composition["top_notes"] + 
                               composition["middle_notes"] + 
                               composition["base_notes"])
                    
                    for note in all_notes[:8]:  # 상위 8개
                        recommendations.append({
                            "name": note["name"],
                            "category": note["category"],
                            "intensity": note["intensity"] * 10,
                            "volatility": note["volatility"],
                            "longevity": note["longevity"],
                            "mood_match": note["mood_score"]
                        })
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        "recommendations": recommendations,
                        "composition": composition,
                        "development_timeline": simulation_result.get("development_timeline", {}),
                        "processing_time": processing_time,
                        "simulation_iterations": 150,
                        "system_mode": "STANDALONE_SIMULATOR",
                        "scene_analysis": {
                            "detected_emotions": emotions,
                            "scene_type": scene_type,
                            "requirements": simulation_result["scene_analysis"]["requirements"]
                        }
                    }
            
            # 기본 시뮬레이터 사용 (폴백)
            elif hasattr(scent_simulator, 'simulate_best_composition'):
                composition, sim_results = scent_simulator.simulate_best_composition(
                    description, scene_type, emotions, iterations=100
                )
                
                composition_dict = scent_simulator.composition_to_dict(composition)
                
                recommendations = []
                all_notes = composition.top_notes + composition.middle_notes + composition.base_notes
                for note in all_notes[:6]:
                    recommendations.append({
                        "name": note.name,
                        "category": note.category,
                        "intensity": note.intensity * 10,
                        "volatility": note.volatility,
                        "longevity": note.longevity,
                        "mood_match": note.mood_score
                    })
                
                processing_time = time.time() - start_time
                
                return {
                    "recommendations": recommendations,
                    "composition": composition_dict,
                    "processing_time": processing_time,
                    "simulation_iterations": 100,
                    "system_mode": "BASIC_SIMULATOR",
                    "scene_analysis": {
                        "detected_emotions": emotions,
                        "scene_type": scene_type,
                        "overall_mood": composition.mood_match,
                        "complexity": composition.complexity
                    }
                }
        
        # 폴백: 시뮬레이터가 없는 경우
        else:
            base_intensity = 50 + (len(description) % 50)
            volatility = "high" if "뜨거운" in description or "강한" in description else "medium"
            
            recommendations = [
                {"name": "시네마틱 블렌드", "intensity": base_intensity, "volatility": volatility},
                {"name": "무비 매직", "intensity": base_intensity + 10, "volatility": "medium"}
            ]
            
            return {"recommendations": recommendations}
        
    except Exception as e:
        logger.error(f"향수 추천 실패: {e}")
        return {"error": f"처리 중 오류: {str(e)}", "recommendations": []}

def _generate_raw_material_formula(raw_materials: list, scent_profile: dict) -> dict:
    """향료 원료 조합 공식 생성"""
    formula = {
        "concentration_ratios": {},
        "mixing_sequence": [],
        "extraction_methods": {},
        "dilution_process": {},
        "aging_requirements": {}
    }
    
    # 농도 비율 계산
    total_concentration = sum(mat.get("concentration", 0) for mat in raw_materials)
    if total_concentration > 0:
        for material in raw_materials:
            name = material["name"]
            concentration = material.get("concentration", 0)
            ratio_percentage = (concentration / total_concentration) * 100
            formula["concentration_ratios"][name] = f"{ratio_percentage:.2f}%"
    
    # 휘발성에 따른 혼합 순서
    base_notes = [mat for mat in raw_materials if mat.get("volatility") == "base"]
    middle_notes = [mat for mat in raw_materials if mat.get("volatility") == "middle"]  
    top_notes = [mat for mat in raw_materials if mat.get("volatility") == "top"]
    
    formula["mixing_sequence"] = [
        "1단계: 베이스노트 혼합 - " + ", ".join([mat["name"] for mat in base_notes]),
        "2단계: 미들노트 추가 - " + ", ".join([mat["name"] for mat in middle_notes]),
        "3단계: 탑노트 블렌딩 - " + ", ".join([mat["name"] for mat in top_notes]),
        "4단계: 에탄올/캐리어 오일 희석",
        "5단계: 숙성 및 안정화"
    ]
    
    # 추출 방법별 분류
    for material in raw_materials:
        method = material.get("extraction_method", "unknown")
        if method not in formula["extraction_methods"]:
            formula["extraction_methods"][method] = []
        formula["extraction_methods"][method].append(material["name"])
    
    # 희석 과정
    total_oils = len(raw_materials)
    if total_oils > 0:
        formula["dilution_process"] = {
            "essential_oils": f"{total_oils} 종류",
            "carrier_ratio": "에센셜 오일 20% : 에탄올 75% : 정제수 5%",
            "final_concentration": f"{scent_profile.get('intensity', 7)}% 향료 농도",
            "recommended_volume": "50ml 기준"
        }
    
    # 숙성 조건
    intensity = scent_profile.get("intensity", 7)
    aging_weeks = max(2, min(8, int(intensity)))
    
    formula["aging_requirements"] = {
        "duration": f"{aging_weeks}주",
        "temperature": "15-20°C",
        "humidity": "50-60%",
        "light_exposure": "어두운 곳 보관",
        "container": "갈색 유리병 사용"
    }
    
    return formula

@app.get("/api/scenes/search")
async def search_scenes(
    emotion: Optional[str] = None,
    scene_type: Optional[str] = None,
    limit: int = 20
):
    """장면 검색 API"""
    if not data_manager:
        raise HTTPException(status_code=503, detail="데이터 매니저가 초기화되지 않았습니다")
    
    try:
        if emotion:
            scenes = data_manager.search_scenes_by_emotion(emotion)
        elif scene_type:
            scenes = data_manager.search_scenes_by_type(scene_type)
        else:
            # 전체 장면 반환 (제한적)
            scenes = list(data_manager.scene_cache.values())
        
        # 제한 적용
        scenes = scenes[:limit]
        
        return {
            "scenes": [scene.to_dict() for scene in scenes],
            "total": len(scenes),
            "query_time": time.time()
        }
        
    except Exception as e:
        logger.error(f"장면 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"검색 처리 중 오류 발생: {str(e)}")

@app.post("/api/movie_capsule")
async def create_movie_capsule(request: dict):
    """🎬 영화용 캡슐 방향제 제조 API"""
    start_time = time.time()
    
    try:
        scene_description = request.get("scene_description", "")
        target_duration = request.get("target_duration", 7.0)  # 기본 7초
        
        if not scene_description.strip():
            raise HTTPException(status_code=400, detail="장면 설명을 입력해주세요")
        
        if not (3 <= target_duration <= 10):
            raise HTTPException(status_code=400, detail="지속시간은 3-10초 사이여야 합니다")
        
        # 캡슐 제조기 초기화
        from core.movie_capsule_formulator import get_capsule_formulator
        formulator = get_capsule_formulator()
        
        # 캡슐 공식 생성
        formula = formulator.formulate_capsule(scene_description, target_duration)
        
        # 상세 보고서 생성
        detailed_report = formulator.generate_detailed_report(formula)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "scene_description": formula.scene_description,
            "target_duration_seconds": formula.target_duration,
            "diffusion_control": formula.diffusion_control,
            "raw_materials": formula.raw_materials,
            "mixing_ratios": formula.mixing_ratios,
            "production_sequence": formula.production_sequence,
            "encapsulation_method": formula.encapsulation_method,
            "activation_mechanism": formula.activation_mechanism,
            "estimated_cost_per_unit": formula.estimated_cost_per_unit,
            "detailed_manufacturing_report": detailed_report,
            "processing_time": processing_time,
            "system_mode": "MOVIE_CAPSULE_FORMULATOR",
            "ml_enhanced": formulator.ml_available
        }
        
    except Exception as e:
        logger.error(f"캡슐 제조 공식 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"캡슐 제조 실패: {str(e)}")

@app.post("/api/video_upload")
async def upload_video_for_analysis(file: UploadFile = File(...), request: Request = None):
    """🎬 비디오 파일 업로드 및 장면 분석 API (보안 강화됨)"""
    if not video_analyzer:
        raise HTTPException(status_code=503, detail="비디오 분석기가 초기화되지 않았습니다")
    
    # 파일 검증을 위해 내용 읽기
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"파일 읽기 실패: {e}")
        raise HTTPException(status_code=400, detail="파일을 읽을 수 없습니다")
    
    # 보안 검증 모듈 import
    try:
        from utils.file_security import (
            comprehensive_video_validation, 
            create_secure_temp_path,
            check_suspicious_patterns,
            log_upload_attempt
        )
    except ImportError:
        logger.warning("보안 모듈을 찾을 수 없습니다. 기본 검증만 사용합니다.")
        # 기본 검증
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="지원하지 않는 비디오 형식입니다")
        if len(contents) > config.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"파일이 너무 큽니다 (최대 {config.max_file_size_mb}MB)")
    else:
        # 클라이언트 IP 추출
        client_ip = request.client.host if request else "unknown"
        
        # 종합 보안 검증
        is_valid, validation_message = comprehensive_video_validation(file.filename, contents)
        if not is_valid:
            log_upload_attempt(file.filename, len(contents), client_ip, success=False)
            raise HTTPException(status_code=400, detail=validation_message)
        
        # 의심스러운 패턴 검사
        is_clean, pattern_message = check_suspicious_patterns(contents)
        if not is_clean:
            log_upload_attempt(file.filename, len(contents), client_ip, success=False)
            raise HTTPException(status_code=400, detail=f"보안 검사 실패: {pattern_message}")
        
        # 성공적인 업로드 로깅
        log_upload_attempt(file.filename, len(contents), client_ip, success=True)
        logger.info(f"보안 검증 완료: {file.filename} - {validation_message}")
    
    try:
        # 보안 강화된 임시 파일 생성
        if 'create_secure_temp_path' in locals():
            temp_path = create_secure_temp_path(file.filename)
            with open(temp_path, 'wb') as temp_file:
                temp_file.write(contents)
            temp_video_path = str(temp_path)
        else:
            # 폴백: 기본 임시 파일 생성
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                temp_file.write(contents)
                temp_video_path = temp_file.name
        
        logger.info(f"보안 검증된 비디오 업로드 완료: {file.filename} ({len(contents)} bytes)")
        
        # 비디오 분석 실행
        start_time = time.time()
        analysis_result = analyze_video_for_scent(temp_video_path, max_frames=config.video_analysis_max_frames)
        processing_time = time.time() - start_time
        
        # 임시 파일 정리
        os.unlink(temp_video_path)
        
        if not analysis_result["success"]:
            raise HTTPException(status_code=500, detail=f"비디오 분석 실패: {analysis_result['error']}")
        
        # 장면 세그먼트를 UI 친화적 형태로 변환
        scene_options = []
        for i, segment in enumerate(analysis_result["scene_segments"]):
            scene_options.append({
                "segment_id": i,
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration": segment["end_time"] - segment["start_time"],
                "primary_mood": segment["summary"]["primary_mood"],
                "dominant_elements": segment["summary"]["dominant_elements"],
                "brightness_level": segment["summary"]["average_brightness"],
                "preview_scent": {
                    "primary_notes": segment["scent"]["scent_profile"]["primary_notes"][:3],
                    "mood": segment["scent"]["scent_profile"]["mood"],
                    "intensity": segment["scent"]["scent_profile"]["intensity"]
                }
            })
        
        return {
            "success": True,
            "filename": file.filename,
            "total_duration": scene_options[-1]["end_time"] if scene_options else 0,
            "frames_analyzed": analysis_result["total_frames_analyzed"],
            "scene_segments": scene_options,
            "overall_mood": analysis_result["overall_scent"]["scene_analysis"]["primary_mood"],
            "processing_time": processing_time,
            "analysis_method": "computer_vision"
        }
        
    except Exception as e:
        # 임시 파일 정리 (에러 시에도)
        try:
            if 'temp_video_path' in locals():
                os.unlink(temp_video_path)
        except:
            pass
        
        logger.error(f"비디오 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"비디오 처리 중 오류 발생: {str(e)}")

@app.post("/api/video_scene_select")
async def select_video_scene_for_scent(request: dict):
    """🎯 특정 비디오 장면 선택 및 향 생성 API"""
    try:
        video_file_path = request.get("video_path", "")
        start_time = request.get("start_time", 0.0)
        end_time = request.get("end_time", 10.0)
        max_frames = request.get("max_frames", 20)
        
        if not video_file_path:
            raise HTTPException(status_code=400, detail="비디오 파일 경로가 필요합니다")
        
        if not video_analyzer:
            raise HTTPException(status_code=503, detail="비디오 분석기가 초기화되지 않았습니다")
        
        # 특정 구간만 분석하는 기능 (추후 구현 가능)
        # 현재는 전체 비디오 분석 후 해당 구간 필터링
        logger.info(f"비디오 구간 분석: {start_time}s - {end_time}s")
        
        analysis_result = analyze_video_for_scent(video_file_path, max_frames)
        
        if not analysis_result["success"]:
            raise HTTPException(status_code=500, detail=f"비디오 분석 실패: {analysis_result['error']}")
        
        # 해당 시간 구간의 세그먼트 찾기
        selected_segment = None
        for segment in analysis_result["scene_segments"]:
            if (segment["start_time"] <= start_time <= segment["end_time"] or
                segment["start_time"] <= end_time <= segment["end_time"] or
                (start_time <= segment["start_time"] and end_time >= segment["end_time"])):
                selected_segment = segment
                break
        
        if not selected_segment:
            # 가장 가까운 세그먼트 사용
            selected_segment = min(analysis_result["scene_segments"], 
                                 key=lambda s: abs(s["start_time"] - start_time))
        
        return {
            "success": True,
            "selected_timerange": {"start": start_time, "end": end_time},
            "matched_segment": {
                "start_time": selected_segment["start_time"],
                "end_time": selected_segment["end_time"],
                "scene_summary": selected_segment["summary"]
            },
            "recommended_scent": selected_segment["scent"],
            "analysis_confidence": selected_segment["scent"]["confidence"],
            "scene_mood": selected_segment["summary"]["primary_mood"],
            "visual_elements": selected_segment["summary"]["dominant_elements"]
        }
        
    except Exception as e:
        logger.error(f"비디오 장면 선택 실패: {e}")
        raise HTTPException(status_code=500, detail=f"장면 선택 처리 중 오류: {str(e)}")

@app.post("/api/multimodal_prediction")
async def multimodal_scent_prediction(request: dict):
    """🎬📝 멀티모달 향수 예측 API (텍스트 + 비디오)"""
    
    if not multimodal_predictor:
        raise HTTPException(status_code=503, detail="멀티모달 예측기가 초기화되지 않았습니다")
    
    try:
        video_path = request.get("video_path")
        text_description = request.get("text_description")
        timerange = request.get("timerange")  # [start, end] in seconds
        
        if not video_path and not text_description:
            raise HTTPException(status_code=400, detail="비디오 파일 또는 텍스트 설명 중 하나는 필수입니다")
        
        # 시간 범위 처리
        selected_timerange = None
        if timerange and len(timerange) == 2:
            selected_timerange = (float(timerange[0]), float(timerange[1]))
        
        start_time = time.time()
        
        # 멀티모달 예측 실행
        prediction_result = multimodal_predictor.predict_from_video_and_text(
            video_path=video_path,
            text_description=text_description,
            selected_timerange=selected_timerange
        )
        
        processing_time = time.time() - start_time
        
        if not prediction_result.get('success'):
            raise HTTPException(status_code=500, detail="멀티모달 예측에 실패했습니다")
        
        combined_pred = prediction_result['combined_prediction']
        
        # API 응답 형식으로 변환
        response = {
            'success': True,
            'prediction_mode': combined_pred.get('analysis_mode', 'multimodal'),
            
            # 향수 프로필
            'scent_profile': {
                'name': combined_pred.get('name', '멀티모달 향수'),
                'description': combined_pred.get('description', ''),
                'intensity': combined_pred.get('intensity', 5.0),
                'confidence': combined_pred.get('confidence', 0.5),
                'predicted_gender': combined_pred.get('predicted_gender', 'unisex')
            },
            
            # 향료 노트
            'fragrance_notes': combined_pred.get('recommended_notes', {}),
            
            # 분석 정보
            'analysis_details': {
                'visual_mood': combined_pred.get('visual_mood', 'neutral'),
                'text_emotions': combined_pred.get('text_emotions', {}),
                'weights': combined_pred.get('weights', {}),
                'source_confidences': combined_pred.get('source_confidences', {})
            },
            
            # 원본 분석 결과
            'raw_analysis': {
                'text_analysis': prediction_result.get('text_analysis', {}),
                'video_analysis': prediction_result.get('video_analysis', {})
            },
            
            # 메타데이터
            'processing_time': processing_time,
            'ml_enhanced': combined_pred.get('ml_enhanced', False),
            'input_sources': {
                'has_video': video_path is not None,
                'has_text': text_description is not None,
                'has_timerange': selected_timerange is not None
            }
        }
        
        logger.info(f"Multimodal prediction completed in {processing_time:.3f}s with confidence {combined_pred.get('confidence', 0):.2f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"멀티모달 예측 실패: {e}")
        raise HTTPException(status_code=500, detail=f"멀티모달 예측 처리 중 오류: {str(e)}")

@app.post("/api/enhanced_video_recommendation")  
async def enhanced_video_recommendation(request: dict):
    """🎬⚡ 향상된 실시간 비디오 향수 추천 API"""
    
    if not recommender:
        raise HTTPException(status_code=503, detail="실시간 추천 시스템이 초기화되지 않았습니다")
    
    try:
        video_path = request.get("video_path")
        text_description = request.get("text_description")
        timerange = request.get("timerange")
        use_cache = request.get("use_cache", True)
        
        if not video_path:
            raise HTTPException(status_code=400, detail="비디오 파일 경로가 필요합니다")
        
        # 시간 범위 튜플 변환
        timerange_tuple = None
        if timerange and len(timerange) == 2:
            timerange_tuple = (float(timerange[0]), float(timerange[1]))
        
        # 실시간 추천 시스템의 비디오 기능 사용
        if hasattr(recommender, 'recommend_for_video_scene'):
            recommendation = recommender.recommend_for_video_scene(
                video_path=video_path,
                text_description=text_description,
                timerange=timerange_tuple,
                use_cache=use_cache
            )
            
            return {
                'success': True,
                'scene_analysis': recommendation.get('scene_analysis', {}),
                'scent_profile': recommendation.get('scent_profile', {}),
                'product_recommendations': recommendation.get('product_recommendations', {}),
                'meta': recommendation.get('meta', {}),
                'system_mode': 'REALTIME_VIDEO_ENHANCED'
            }
        else:
            # 폴백: 기존 텍스트 기반 추천 사용
            description_text = text_description or f"비디오 파일 기반 장면 ({video_path})"
            
            recommendation = recommender.recommend_for_scene(
                description=description_text,
                scene_type="auto",
                mood="auto",
                intensity_preference=5
            )
            
            recommendation['meta']['fallback_mode'] = 'text_based'
            recommendation['meta']['original_video_path'] = video_path
            
            return recommendation
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"향상된 비디오 추천 실패: {e}")
        raise HTTPException(status_code=500, detail=f"비디오 추천 처리 중 오류: {str(e)}")

@app.get("/api/stats")
async def get_system_stats():
    """시스템 통계 API"""
    stats = {}
    
    if data_manager:
        stats["data_manager"] = data_manager.get_performance_stats()
    
    if recommender:
        stats["recommender"] = recommender.get_performance_stats()
    
    if video_analyzer:
        stats["video_analyzer"] = {
            "status": "active",
            "supported_formats": video_analyzer.supported_formats
        }
    
    if multimodal_predictor:
        stats["multimodal_predictor"] = {
            "status": "active",
            "text_predictor_available": multimodal_predictor.text_predictor is not None,
            "video_analyzer_available": multimodal_predictor.video_analyzer is not None
        }
    
    return {
        "stats": stats,
        "system_capabilities": {
            "text_analysis": data_manager is not None or recommender is not None,
            "video_analysis": video_analyzer is not None,
            "multimodal_prediction": multimodal_predictor is not None,
            "deep_learning": dl_predictor is not None or trained_predictor is not None,
            "realtime_recommendations": recommender is not None,
            "scent_simulation": scent_simulator is not None
        },
        "timestamp": time.time()
    }

if __name__ == "__main__":
    print(f"Movie Scent AI Web Server Starting...")
    print(f"Web Interface: http://{config.server_host}:{config.server_port}")
    print(f"API Docs: http://{config.server_host}:{config.server_port}/docs")
    print(f"Configuration: {config.to_dict()}")
    
    uvicorn.run(
        "app:app",
        host=config.server_host,
        port=config.server_port,
        reload=config.reload_on_change,
        log_level=config.log_level.lower()
    )