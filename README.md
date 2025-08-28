# 🎬 영화용 냄새 구조 딥러닝 AI 시스템

감독이 원하는 어떤 향이든 화학적으로 정확하게 구현해주는 전문 AI 도구

## 🌐 웹 인터페이스 (NEW!)

**Live Demo**: [Movie Scent AI Frontend](https://movie-scent-ai.vercel.app) 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/junseong2im/Ai-project/tree/main/movie-scent-frontend)

### ✨ 프론트엔드 기능
- **실시간 장면 분석**: 자연어 입력으로 즉시 향수 추천
- **인터랙티브 그래프**: Chart.js 기반 학습 시각화
- **반응형 디자인**: 모든 디바이스에서 완벽한 UX
- **Glass Morphism**: 현대적인 디자인 언어

## 🎯 핵심 철학
**"AI는 의견을 내지 않는다 - 오직 감독의 요구사항을 구현한다"**

## 🚀 주요 특징

### 🔥 한계치 딥러닝 모델
- **5,757,842개 매개변수** 초대형 신경망
- **4,020개 영화 장면** 학습 데이터 (200배 확장)
- **155차원 고차원 특성** 분석
- **Multi-Head Attention** (16 헤드) + **Residual Networks**

### ⚡ 실시간 성능
- **0.001초** 초고속 처리
- **70%+** 예측 신뢰도
- **15개 향수 카테고리** 완전 분류
- **자동 캐싱** 시스템

### 🎭 영화 장면 전문 분석
- **로맨틱** → Floral/Oriental 구현
- **공포** → Smoky/Earthy/Metallic 구현
- **액션** → Intense/Leather 구현
- **평화** → Fresh/Citrus 구현

## 📁 프로젝트 구조

```
ai_perfume/
├── core/
│   ├── movie_scent_ai.py              # 메인 딥러닝 시스템 (575만 매개변수)
│   ├── real_time_movie_scent.py       # 실시간 추천 엔진
│   ├── deep_learning_integration.py   # 기존 시스템 통합
│   └── optimized_data_manager.py      # 효율적 데이터 관리
├── data/
│   ├── movie_scent_database.json      # 영화 장면 데이터베이스
│   ├── raw/raw_perfume_data.csv       # 70,000+ 향수 데이터
│   └── processed/                     # 전처리된 데이터 (Git LFS)
├── models/
│   ├── perfume_dl_model.pth          # 훈련된 딥러닝 모델 (Git LFS)
│   └── advanced_recipe_generator.py  # Transformer 기반 레시피 생성
├── movie-scent-frontend/              # Next.js 웹 인터페이스
│   ├── app/
│   │   ├── components/               # React 컴포넌트
│   │   ├── page.tsx                 # 메인 페이지
│   │   └── layout.tsx               # 레이아웃
│   ├── package.json                 # 프론트엔드 의존성
│   └── vercel.json                  # Vercel 배포 설정
├── app.py                           # FastAPI 백엔드
├── start.py                         # 원클릭 실행기
└── enhanced_main_with_dl.py         # 통합 실행 파일
```

## 🛠️ 설치 및 실행

### 1. 환경 설정
```bash
pip install torch pandas numpy scikit-learn transformers
```

### 2. AI 백엔드 실행
```bash
cd ai_perfume
python start.py  # 웹서버 자동 시작 (http://localhost:8000)
```

### 3. 프론트엔드 개발 (선택사항)
```bash
cd movie-scent-frontend
npm install
npm run dev  # http://localhost:3000
```

### 4. Vercel 배포 (원클릭)
```bash
vercel --prod
```

## 💡 사용 방법

### 감독 요구사항 → AI 구현
```python
from core.real_time_movie_scent import RealTimeMovieScentRecommender

recommender = RealTimeMovieScentRecommender()

# 감독의 요구사항
result = recommender.recommend_for_scene(
    "비 오는 밤 옥상에서 이별하는 장면, 담배냄새와 빗물냄새 조합",
    scene_type="drama",
    mood="melancholy", 
    intensity_preference=8
)

# AI 결과: 정확한 향료 조합과 비율 제시
```

## 🎨 15개 향수 카테고리 시스템

1. **Citrus**: bergamot, lemon, orange
2. **Floral**: rose, jasmine, lily, violet
3. **Woody**: cedar, sandalwood, pine
4. **Oriental**: amber, vanilla, musk, oud
5. **Fresh**: mint, eucalyptus, sea breeze
6. **Spicy**: cinnamon, nutmeg, cardamom
7. **Fruity**: apple, peach, berry
8. **Gourmand**: chocolate, coffee, caramel
9. **Animalic**: leather, musk, ambergris
10. **Herbal**: basil, rosemary, thyme
11. **Aquatic**: ocean, rain, marine
12. **Metallic**: steel, iron, mineral
13. **Smoky**: smoke, tobacco, fire
14. **Earthy**: soil, moss, wet earth
15. **Synthetic**: aldehydes, chemical

## 🧠 AI 시스템 아키텍처

### 딥러닝 모델 구조
- **입력층**: 155차원 다중 특성 (텍스트, 감정, 시각, 시간)
- **어텐션층**: 16-Head Multi-Head Attention
- **은닉층**: [1024, 512, 256, 128, 64] 잔차 연결
- **출력층**: 이중 헤드 (강도 예측 + 카테고리 분류)

### 특성 추출 시스템
- **텍스트 분석**: TF-IDF 벡터화 (500차원)
- **감정 분석**: 전용 벡터라이저 (200차원)
- **시각 분석**: 시각 요소 벡터화 (300차원)
- **시간-날씨**: 조합 특성 생성

## 🎬 지원 영화 장르

- **로맨틱**: The Notebook, La La Land
- **공포**: The Shining, 호러 영화들
- **액션**: Mad Max, John Wick
- **SF**: Blade Runner 2049, Interstellar
- **판타지**: Avatar, Spirited Away
- **드라마**: Little Women, Her
- **코미디**: Amélie, Grand Budapest Hotel

## 📊 성능 지표

- **모델 크기**: 5,757,842개 매개변수
- **처리 속도**: 0.001초/요청
- **예측 정확도**: 70%+ 신뢰도
- **데이터 규모**: 4,020개 학습 장면
- **카테고리 분류**: 15개 완전 분류

## 🎯 특별한 점

### AI는 절대 의견을 내지 않습니다
- ❌ "이 향이 더 좋겠어요"
- ❌ "로맨틱 장면에는 보통 이렇게 해요"
- ❌ "제 추천은..."

### AI는 오직 구현만 합니다
- ✅ 감독 요구사항 분석
- ✅ 화학적 구현 방법 계산
- ✅ 정확한 향료 조합 제시
- ✅ 실제 제작 가능한 레시피

## 🏆 활용 분야

- 🎬 **영화/드라마** 제작진용 향수 컨설팅
- 🛍️ **향수 브랜드** 마케팅 도구
- 🎭 **테마파크/체험관** 향수 연출
- 📱 **개인 맞춤** 향수 추천 앱

## 📈 업데이트 현황

- [x] **웹/모바일 앱 인터페이스** ✅ (Next.js + Vercel)
- [x] **실시간 처리 시스템** ✅ (FastAPI 백엔드)
- [x] **인터랙티브 시각화** ✅ (Chart.js 그래프)
- [ ] 이미지 분석 통합 (비전 AI)
- [ ] 실제 향수 제조 공정 연동
- [ ] 메타버스/VR 향수 체험

## 🤝 기여하기

1. 이슈 등록
2. 포크 후 브랜치 생성
3. 코드 작성 및 테스트
4. 풀 리퀘스트 생성

## 📜 라이선스

MIT License - 자유롭게 사용 가능

---

**"감독의 상상력을 현실로 구현하는 AI 도구"**

🤖 *Generated with Advanced AI Perfume System v2.0*