#!/usr/bin/env python3
"""
영화 향수 시스템 간단 테스트
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))

from core.real_time_movie_scent import RealTimeMovieScentRecommender

def test_movie_scent_system():
    """영화 향수 시스템 테스트"""
    
    print("=== 영화 향수 추천 시스템 테스트 ===")
    
    # 시스템 초기화
    recommender = RealTimeMovieScentRecommender()
    
    # 테스트 시나리오들
    scenarios = [
        {
            'description': "해변에서 석양을 바라보며 와인을 마시는 로맨틱한 장면",
            'scene_type': "romantic",
            'mood': "love",
            'intensity': 6
        },
        {
            'description': "어두운 숲속에서 괴물과 마주치는 공포 장면",
            'scene_type': "horror",
            'mood': "fear", 
            'intensity': 9
        },
        {
            'description': "파리 카페에서 커피를 마시는 평화로운 아침",
            'scene_type': "drama",
            'mood': "peaceful",
            'intensity': 4
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[테스트 {i}] {scenario['scene_type'].upper()}")
        print(f"장면: {scenario['description']}")
        print("-" * 50)
        
        start_time = time.time()
        
        # 추천 생성
        result = recommender.recommend_for_scene(
            scenario['description'],
            scenario['scene_type'],
            scenario['mood'],
            scenario['intensity']
        )
        
        processing_time = time.time() - start_time
        
        # 결과 출력
        scent = result['scent_profile']
        print(f"향수 프로필:")
        print(f"  - 강도: {scent['intensity']:.1f}/10")
        print(f"  - 지속성: {scent['longevity']:.1f}/10") 
        print(f"  - 투사력: {scent['projection']:.1f}/10")
        print(f"  - 카테고리: {', '.join(scent['primary_categories'])}")
        print(f"  - 신뢰도: {scent['confidence']:.1%}")
        
        recommendations = result['product_recommendations']['top_picks'][:3]
        print(f"추천 제품:")
        for j, rec in enumerate(recommendations, 1):
            print(f"  {j}. {rec['brand']} - {rec['name']} ({rec['category']})")
        
        print(f"처리 시간: {processing_time:.3f}초")
    
    # 성능 통계
    stats = recommender.get_performance_stats()
    print(f"\n=== 성능 통계 ===")
    print(f"총 추천 수: {stats['total_recommendations']}")
    print(f"평균 응답 시간: {stats['average_response_time']}")
    print(f"캐시 적중률: {stats['cache_hit_rate']}")
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    test_movie_scent_system()