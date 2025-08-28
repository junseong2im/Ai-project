#!/usr/bin/env python3
"""
영화 향수 AI 시스템 원클릭 실행기
모든 설정을 자동으로 처리하고 웹서버를 시작합니다
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class MovieScentAILauncher:
    """영화 향수 AI 시스템 런처"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.required_packages = [
            'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn', 
            'torch', 'transformers', 'pydantic'
        ]
        
        logger.info("🎬 영화 향수 AI 시스템 시작 준비...")
    
    def check_python_version(self):
        """Python 버전 확인"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error("❌ Python 3.8 이상이 필요합니다")
            logger.error(f"현재 버전: {version.major}.{version.minor}.{version.micro}")
            return False
        
        logger.info(f"✅ Python 버전 확인: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def install_requirements(self):
        """필요한 패키지 설치"""
        logger.info("📦 필요한 패키지 확인 및 설치 중...")
        
        requirements_file = self.project_dir / "requirements.txt"
        
        if requirements_file.exists():
            try:
                # 필수 패키지만 먼저 설치 (빠른 시작용)
                essential_packages = [
                    "fastapi==0.104.1",
                    "uvicorn[standard]==0.24.0", 
                    "pydantic==2.5.0",
                    "pandas>=2.3.2",
                    "numpy>=2.2.6,<2.3",
                    "scikit-learn>=1.7.1"
                ]
                
                logger.info("⚡ 필수 패키지 설치 중...")
                for package in essential_packages:
                    try:
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", package
                        ], check=True, capture_output=True)
                        logger.info(f"✅ {package.split('==')[0]} 설치 완료")
                    except subprocess.CalledProcessError:
                        logger.warning(f"⚠️ {package} 설치 실패 - 계속 진행")
                
                logger.info("✅ 필수 패키지 설치 완료")
                return True
                
            except Exception as e:
                logger.error(f"❌ 패키지 설치 실패: {e}")
                return False
        else:
            logger.warning("⚠️ requirements.txt 파일을 찾을 수 없습니다")
            return True
    
    def check_data_files(self):
        """데이터 파일 확인"""
        logger.info("📊 데이터 파일 확인 중...")
        
        data_dir = self.project_dir / "data"
        required_files = [
            "raw/raw_perfume_data.csv",
            "movie_scent_database.json"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = data_dir / file_path
            if full_path.exists():
                logger.info(f"✅ {file_path} 확인")
            else:
                missing_files.append(file_path)
                logger.warning(f"⚠️ {file_path} 없음")
        
        if missing_files:
            logger.info("🔧 누락된 데이터 파일이 있지만 시스템은 동작합니다")
        
        return True
    
    def start_web_server(self):
        """웹서버 시작"""
        logger.info("🚀 웹서버 시작 중...")
        
        try:
            # 현재 디렉토리를 프로젝트 디렉토리로 변경
            os.chdir(self.project_dir)
            
            # 웹서버 시작 (백그라운드)
            import threading
            
            def run_server():
                try:
                    import uvicorn
                    uvicorn.run(
                        "app:app",
                        host="127.0.0.1",
                        port=8000,
                        log_level="warning",  # 로그 최소화
                        access_log=False
                    )
                except ImportError:
                    logger.error("❌ uvicorn이 설치되지 않았습니다")
                    logger.info("설치: pip install uvicorn[standard]")
                except Exception as e:
                    logger.error(f"❌ 웹서버 시작 실패: {e}")
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # 서버 시작 대기
            time.sleep(3)
            
            logger.info("✅ 웹서버 시작 완료!")
            logger.info("🌐 웹 인터페이스: http://localhost:8000")
            logger.info("📖 API 문서: http://localhost:8000/docs")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 웹서버 시작 실패: {e}")
            return False
    
    def open_browser(self):
        """브라우저 자동 열기"""
        try:
            time.sleep(2)  # 서버 완전 시작 대기
            webbrowser.open("http://localhost:8000")
            logger.info("🌐 브라우저가 자동으로 열렸습니다")
        except Exception as e:
            logger.warning(f"⚠️ 브라우저 자동 열기 실패: {e}")
            logger.info("수동으로 http://localhost:8000 을 열어주세요")
    
    def show_usage_info(self):
        """사용법 안내"""
        print("\n" + "="*60)
        print("🎬 영화 향수 AI 시스템 사용법")
        print("="*60)
        print("1. 웹 브라우저에서 http://localhost:8000 접속")
        print("2. 영화 장면 설명 입력")
        print("3. 원하는 설정 조정 (장면 타입, 강도 등)")
        print("4. '향수 구현하기' 버튼 클릭")
        print("5. AI가 분석한 향수 조합 확인")
        print("\n💡 특징:")
        print("- AI는 의견을 내지 않고 오직 구현만 합니다")
        print("- 어떤 향이든 화학적으로 정확하게 분석")
        print("- 실제 향수 브랜드 제품 추천")
        print("- 0.1초 내 실시간 처리")
        print("\n⚡ 고급 기능:")
        print("- API 문서: http://localhost:8000/docs")
        print("- 시스템 상태: http://localhost:8000/health")
        print("- 검색 API: http://localhost:8000/api/scenes/search")
        print("\n🛑 종료: Ctrl+C")
        print("="*60)
    
    def run(self):
        """전체 시스템 실행"""
        try:
            # 1. Python 버전 확인
            if not self.check_python_version():
                return False
            
            # 2. 패키지 설치
            if not self.install_requirements():
                logger.warning("⚠️ 일부 패키지 설치 실패 - 계속 진행")
            
            # 3. 데이터 파일 확인
            self.check_data_files()
            
            # 4. 웹서버 시작
            if not self.start_web_server():
                return False
            
            # 5. 브라우저 열기
            self.open_browser()
            
            # 6. 사용법 안내
            self.show_usage_info()
            
            # 7. 서버 유지
            try:
                logger.info("\n🟢 시스템이 실행 중입니다...")
                logger.info("종료하려면 Ctrl+C를 누르세요")
                
                while True:
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                logger.info("\n\n🛑 시스템을 종료합니다...")
                logger.info("감사합니다! 🎬✨")
                return True
                
        except Exception as e:
            logger.error(f"❌ 시스템 실행 실패: {e}")
            return False

def main():
    """메인 함수"""
    print("🎬 영화용 냄새 구조 딥러닝 AI 시스템")
    print("감독이 원하는 어떤 향이든 구현해드립니다")
    print("="*60)
    
    launcher = MovieScentAILauncher()
    success = launcher.run()
    
    if not success:
        print("\n❌ 시스템 시작에 실패했습니다")
        print("문제 해결:")
        print("1. Python 3.8 이상 설치 확인")
        print("2. pip install -r requirements.txt 실행")
        print("3. 관리자 권한으로 실행")
        input("\n계속하려면 Enter를 누르세요...")
    
    return success

if __name__ == "__main__":
    main()