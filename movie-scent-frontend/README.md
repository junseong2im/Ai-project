# 🎬 Movie Scent AI Frontend

> 감독이 원하는 어떤 향이든 AI가 구현해드립니다

영화 장면을 입력하면 AI가 최적의 향수 조합을 분석하고 추천하는 혁신적인 웹 애플리케이션입니다.

## ✨ 주요 기능

### 🎯 **장면 분석**
- 자연어로 영화 장면 설명 입력
- 7가지 장면 타입 자동 분류 (로맨틱, 공포, 액션, 드라마, 코미디, SF, 판타지)
- 시간대, 위치, 강도 설정
- 실시간 예시 장면 제공

### 🧪 **AI 분석 결과**
- **화학적 구조 분석**: Top/Middle/Base Notes 상세 분석
- **성능 지표**: 강도, 지속성, 투사력, 신뢰도
- **카테고리 매칭**: 15개 향수 카테고리 중 최적 선택
- **브랜드 추천**: 실제 향수 제품 매칭 점수와 함께

### 📊 **학습 시각화**
- 신경망 아키텍처 구조도
- 실시간 성능 지표 애니메이션
- 장면별 정확도 분석
- 5,757,842개 파라미터 딥러닝 모델 성능

## 🚀 기술 스택

### Frontend
- **Next.js 14** - React 프레임워크
- **TypeScript** - 타입 안정성
- **Tailwind CSS** - 스타일링
- **Chart.js** - 데이터 시각화
- **Lucide React** - 아이콘

### AI Backend (연동)
- **PyTorch** - 딥러닝 프레임워크
- **Multi-Head Attention** - 16 헤드
- **Residual Networks** - Skip Connections
- **5,757,842 Parameters** - 대규모 신경망

## 🎨 디자인 특징

- **Glass Morphism** - 현대적인 유리 효과
- **Gradient Backgrounds** - 아름다운 그래디언트
- **Animated Statistics** - 실시간 숫자 애니메이션
- **Responsive Design** - 모든 디바이스 지원
- **Interactive Charts** - 동적 데이터 시각화

## 🏃‍♂️ 빠른 시작

### 로컬 개발 환경

```bash
# 저장소 클론
git clone https://github.com/junseong2im/Ai-project
cd movie-scent-frontend

# 의존성 설치
npm install

# 개발 서버 시작
npm run dev
```

### Vercel 배포

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/junseong2im/Ai-project/tree/main/movie-scent-frontend)

1. Vercel 계정으로 로그인
2. GitHub 저장소 연결
3. 자동 배포 완료!

## 📱 사용법

### 1단계: 장면 입력
```
예시: "비 오는 밤 옥상에서 이별하는 장면, 
담배냄새와 빗물냄새가 섞인 쓸쓸한 분위기"
```

### 2단계: 설정 조정
- **장면 타입**: 로맨틱, 공포, 액션 등 선택
- **시간대**: 새벽, 아침, 오후, 저녁, 밤
- **위치**: 옥상, 해변, 숲속 등 (선택사항)
- **강도**: 1(은은함) ~ 10(강렬함)

### 3단계: AI 분석 결과 확인
- 화학적 구조 프로필
- 성능 지표 (강도, 지속성, 투사력, 신뢰도)
- 추천 브랜드 제품 목록
- 매칭 점수 및 설명

### 4단계: 학습 그래프 확인
- 신경망 아키텍처 구조
- 훈련 진행 곡선
- 장면별 성능 분석

## 🎯 AI 모델 성능

| 지표 | 값 |
|------|-----|
| **전체 정확도** | 94.7% |
| **평균 처리 시간** | 0.08초 |
| **신경망 파라미터** | 5,757,842개 |
| **학습 장면 수** | 4,020개 |
| **향수 데이터베이스** | 70,103개 제품 |

## 🔧 개발 스크립트

```bash
# 개발 서버 시작
npm run dev

# 프로덕션 빌드
npm run build

# 정적 파일 생성
npm run export

# 린터 실행
npm run lint
```

## 📂 프로젝트 구조

```
movie-scent-frontend/
├── app/
│   ├── components/
│   │   ├── SceneInputForm.tsx      # 장면 입력 폼
│   │   ├── ResultsDisplay.tsx      # 결과 표시
│   │   └── TrainingVisualization.tsx # 학습 그래프
│   ├── globals.css                 # 글로벌 스타일
│   ├── layout.tsx                  # 루트 레이아웃
│   └── page.tsx                   # 메인 페이지
├── public/                        # 정적 파일
├── next.config.js                 # Next.js 설정
├── tailwind.config.js            # Tailwind 설정
├── vercel.json                   # Vercel 배포 설정
└── package.json                  # 프로젝트 설정
```

## 🌐 라이브 데모

**Vercel**: [https://movie-scent-ai.vercel.app](https://movie-scent-ai.vercel.app)

## 🤝 기여하기

1. Fork 저장소
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👨‍💻 개발자

**Movie Scent AI Team**
- GitHub: [@junseong2im](https://github.com/junseong2im)
- 프로젝트: [Ai-project](https://github.com/junseong2im/Ai-project)

---

*감독이 원하는 어떤 향이든 AI가 구현해드립니다* ✨