#!/usr/bin/env python3
"""
영화 향수 AI 웹 애플리케이션
FastAPI + React-like interface
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
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

# 우리 시스템 임포트
try:
    from core.optimized_data_manager import get_data_manager, SceneData
    from core.real_time_movie_scent import RealTimeMovieScentRecommender
    from core.deep_learning_integration import DeepLearningPerfumePredictor, get_trained_predictor
    from core.scent_simulator import ScentSimulator
    from core.standalone_scent_simulator import StandaloneScentSimulator
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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="영화 향수 AI API",
    description="감독이 원하는 어떤 향이든 구현해주는 전문 AI 시스템",
    version="2.0.0"
)

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    global data_manager, recommender, dl_predictor, trained_predictor, scent_simulator
    
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
                mixing_formula = self._generate_raw_material_formula(raw_materials, scent_profile)
                
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

@app.get("/api/stats")
async def get_system_stats():
    """시스템 통계 API"""
    stats = {}
    
    if data_manager:
        stats["data_manager"] = data_manager.get_performance_stats()
    
    if recommender:
        stats["recommender"] = recommender.get_performance_stats()
    
    return {
        "stats": stats,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    print("Movie Scent AI Web Server Starting...")
    print("Web Interface: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )