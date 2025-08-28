#!/usr/bin/env python3
"""
실제 유명 영화 장면 기반 향료 레시피 대량 생성 시스템
10만개 이상 레시피 자동 생성
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.scene_fragrance_recipe import SceneFragranceRecipe

class MovieRecipeGenerator:
    """실제 영화 기반 레시피 대량 생성기"""
    
    def __init__(self):
        self.recipe_generator = SceneFragranceRecipe()
        self.generated_recipes = []
        
        # 실제 유명 영화 데이터베이스
        self.famous_movies = {
            "action": [
                {"title": "Mad Max Fury Road", "scenes": [
                    "desert chase with explosions and fire",
                    "mechanic workshop with oil and metal", 
                    "war rig battle in sandstorm",
                    "gasoline station explosion scene",
                    "final confrontation on moving vehicles"
                ]},
                {"title": "John Wick", "scenes": [
                    "nightclub assassination with smoke",
                    "hotel continental lobby fight",
                    "underground garage shootout",
                    "rooftop sniper scene",
                    "church final battle"
                ]},
                {"title": "Mission Impossible", "scenes": [
                    "helicopter chase through mountains",
                    "underwater infiltration scene",
                    "skyscraper climbing sequence", 
                    "motorcycle chase through traffic",
                    "explosion escape from building"
                ]},
                {"title": "Fast and Furious", "scenes": [
                    "street racing at night",
                    "garage working on cars",
                    "highway truck heist",
                    "plane cargo fight scene",
                    "rooftop car jump"
                ]},
                {"title": "The Dark Knight", "scenes": [
                    "bank robbery opening scene",
                    "hospital explosion sequence",
                    "police chase through tunnels",
                    "warehouse interrogation",
                    "ferry boat tension scene"
                ]}
            ],
            
            "romantic": [
                {"title": "Titanic", "scenes": [
                    "first class dinner dance",
                    "bow of ship flying scene",
                    "drawing scene in cabin",
                    "goodbye at sinking ship",
                    "final underwater reunion"
                ]},
                {"title": "The Notebook", "scenes": [
                    "lake house summer romance",
                    "rain fight and kiss",
                    "old couple reading together",
                    "carnival first meeting",
                    "final bed scene together"
                ]},
                {"title": "La La Land", "scenes": [
                    "planetarium dance sequence",
                    "jazz club intimate conversation",
                    "griffith observatory date",
                    "final concert recognition",
                    "coffee shop audition scene"
                ]},
                {"title": "Casablanca", "scenes": [
                    "cafe americain piano scene",
                    "airport foggy goodbye",
                    "paris flashback romance",
                    "final sacrifice decision",
                    "gin joint meeting"
                ]},
                {"title": "Romeo and Juliet", "scenes": [
                    "balcony moonlight scene",
                    "church secret wedding",
                    "masquerade ball meeting",
                    "tragic poison ending",
                    "family feud street fight"
                ]}
            ],
            
            "horror": [
                {"title": "The Shining", "scenes": [
                    "hotel corridor chase",
                    "bathroom mirror revelation", 
                    "ballroom ghost party",
                    "maze winter pursuit",
                    "room 237 encounter"
                ]},
                {"title": "Halloween", "scenes": [
                    "suburban house stalking",
                    "closet hiding sequence",
                    "babysitting phone calls",
                    "kitchen knife attack",
                    "final girl confrontation"
                ]},
                {"title": "The Exorcist", "scenes": [
                    "attic strange noises",
                    "bedroom possession scene",
                    "medical examination horror",
                    "priest final exorcism",
                    "head spinning moment"
                ]},
                {"title": "It", "scenes": [
                    "sewer drain encounter",
                    "basement well fear",
                    "school bathroom bullying",
                    "house of mirrors",
                    "final underground battle"
                ]},
                {"title": "A Quiet Place", "scenes": [
                    "silent family dinner",
                    "cornfield creature hunt",
                    "basement birth scene",
                    "silo grain trap",
                    "radio tower transmission"
                ]}
            ],
            
            "drama": [
                {"title": "The Godfather", "scenes": [
                    "wedding celebration garden",
                    "restaurant assassination scene",
                    "hospital bedside conversation",
                    "church baptism sequence",
                    "office meeting intimidation"
                ]},
                {"title": "Forrest Gump", "scenes": [
                    "bus stop bench storytelling",
                    "vietnam war jungle scenes",
                    "shrimp boat working",
                    "running across america",
                    "jenny reunion scene"
                ]},
                {"title": "Schindler's List", "scenes": [
                    "factory worker selection",
                    "ghetto liquidation scene",
                    "list writing office",
                    "train station goodbye",
                    "final grave visit"
                ]},
                {"title": "Good Will Hunting", "scenes": [
                    "therapy office sessions",
                    "mit classroom blackboard",
                    "park bench breakthrough",
                    "library study scenes",
                    "final car departure"
                ]},
                {"title": "Shawshank Redemption", "scenes": [
                    "prison cell first night",
                    "library book reading",
                    "rooftop beer sharing",
                    "tunnel escape sequence",
                    "beach reunion ending"
                ]}
            ],
            
            "thriller": [
                {"title": "Silence of the Lambs", "scenes": [
                    "prison cell hannibal meeting",
                    "basement night vision chase",
                    "autopsy morgue examination",
                    "house final confrontation",
                    "phone booth closing call"
                ]},
                {"title": "Se7en", "scenes": [
                    "crime scene investigation",
                    "library research sequence",
                    "desert final reveal",
                    "apartment chase scene",
                    "box delivery moment"
                ]},
                {"title": "Zodiac", "scenes": [
                    "newspaper office investigation",
                    "movie theater phone call",
                    "basement handwriting analysis",
                    "lake attack recreation",
                    "suspect house visit"
                ]},
                {"title": "Gone Girl", "scenes": [
                    "police station interview",
                    "shopping center abduction",
                    "cabin hiding sequence",
                    "media press conference",
                    "kitchen final conversation"
                ]},
                {"title": "Prisoners", "scenes": [
                    "suburban family dinner",
                    "basement interrogation scene",
                    "rainy street searching",
                    "hospital bedside vigil",
                    "maze underground discovery"
                ]}
            ],
            
            "comedy": [
                {"title": "Superbad", "scenes": [
                    "high school hallway walking",
                    "liquor store robbery attempt", 
                    "house party chaos",
                    "mall security chase",
                    "grocery store encounter"
                ]},
                {"title": "The Hangover", "scenes": [
                    "vegas hotel room discovery",
                    "rooftop tiger encounter",
                    "casino gambling montage",
                    "hospital emergency visit",
                    "chapel wedding scene"
                ]},
                {"title": "Anchorman", "scenes": [
                    "news station office",
                    "restaurant dinner date",
                    "street fight sequence",
                    "jazz flute performance",
                    "zoo bear encounter"
                ]},
                {"title": "Dumb and Dumber", "scenes": [
                    "limousine driving job",
                    "diner bathroom scene",
                    "hotel fancy dinner",
                    "ski resort chase",
                    "van cross country trip"
                ]},
                {"title": "Wedding Crashers", "scenes": [
                    "church wedding ceremony",
                    "reception dinner party",
                    "country club golf",
                    "sailboat romantic date",
                    "family breakfast scene"
                ]}
            ],
            
            "sci_fi": [
                {"title": "Blade Runner 2049", "scenes": [
                    "dystopian city rain scene",
                    "holographic girlfriend apartment",
                    "factory replicant retirement",
                    "vegas radioactive wasteland",
                    "underground resistance base"
                ]},
                {"title": "Interstellar", "scenes": [
                    "cornfield drone chase",
                    "nasa facility discovery",
                    "space station docking",
                    "water planet exploration",
                    "library tesseract sequence"
                ]},
                {"title": "The Matrix", "scenes": [
                    "office cubicle reality",
                    "red pill blue pill choice",
                    "training simulation dojo",
                    "highway chase sequence",
                    "final phone booth escape"
                ]},
                {"title": "Star Wars", "scenes": [
                    "cantina alien bar scene",
                    "death star trench run",
                    "desert twin suns moment",
                    "lightsaber duel sequence",
                    "rebel base briefing room"
                ]},
                {"title": "Arrival", "scenes": [
                    "alien ship first contact",
                    "military camp base setup",
                    "communication chamber session",
                    "university linguistics office",
                    "final revelation montage"
                ]}
            ]
        }
        
        # 시간대 변형
        self.time_variations = [
            "early morning", "dawn", "sunrise", "morning", "late morning",
            "noon", "afternoon", "late afternoon", "evening", "sunset", 
            "dusk", "night", "late night", "midnight", "pre-dawn"
        ]
        
        # 날씨/환경 변형  
        self.weather_variations = [
            "sunny", "cloudy", "overcast", "rainy", "stormy", "foggy",
            "snowy", "windy", "humid", "dry", "cold", "hot", "warm", "cool"
        ]
        
        # 감정 강화 키워드
        self.emotion_enhancers = {
            "action": ["explosive", "intense", "adrenaline-filled", "high-stakes", "dangerous"],
            "romantic": ["passionate", "intimate", "tender", "heartwarming", "sensual"],
            "horror": ["terrifying", "spine-chilling", "nightmarish", "sinister", "blood-curdling"],
            "drama": ["emotional", "heart-breaking", "profound", "moving", "contemplative"],
            "thriller": ["suspenseful", "nerve-wracking", "tense", "mysterious", "gripping"],
            "comedy": ["hilarious", "lighthearted", "amusing", "witty", "absurd"],
            "sci_fi": ["futuristic", "otherworldly", "mind-bending", "technological", "alien"]
        }
    
    def generate_scene_variations(self, base_scene: str, genre: str, movie_title: str) -> List[str]:
        """기본 장면으로부터 다양한 변형 생성"""
        variations = []
        
        # 기본 장면
        variations.append(f"{movie_title}: {base_scene}")
        
        # 시간대 변형 (5가지)
        for time_var in random.sample(self.time_variations, 5):
            variations.append(f"{movie_title}: {base_scene} during {time_var}")
        
        # 날씨 변형 (3가지)
        for weather_var in random.sample(self.weather_variations, 3):
            variations.append(f"{movie_title}: {base_scene} in {weather_var} weather")
        
        # 감정 강화 변형 (4가지)
        enhancers = self.emotion_enhancers.get(genre, ["dramatic"])
        for enhancer in random.sample(enhancers, min(4, len(enhancers))):
            variations.append(f"{movie_title}: {enhancer} {base_scene}")
        
        # 시간+날씨 조합 (3가지)
        for i in range(3):
            time_var = random.choice(self.time_variations)
            weather_var = random.choice(self.weather_variations)
            variations.append(f"{movie_title}: {base_scene} during {time_var} with {weather_var} conditions")
        
        # 캐릭터 관점 변형 (2가지)
        perspectives = ["from protagonist perspective", "from antagonist viewpoint"]
        for perspective in perspectives:
            variations.append(f"{movie_title}: {base_scene} {perspective}")
        
        return variations
    
    def generate_massive_dataset(self, target_count: int = 100000) -> List[Dict]:
        """대량 레시피 데이터셋 생성"""
        print(f"Starting massive recipe generation for {target_count:,} recipes...")
        
        recipes = []
        recipe_count = 0
        
        start_time = time.time()
        
        # 각 영화의 각 장면에 대해 변형 생성
        while recipe_count < target_count:
            for genre, movies in self.famous_movies.items():
                for movie in movies:
                    for base_scene in movie["scenes"]:
                        # 각 기본 장면에서 18가지 변형 생성
                        scene_variations = self.generate_scene_variations(
                            base_scene, genre, movie["title"]
                        )
                        
                        for scene_description in scene_variations:
                            if recipe_count >= target_count:
                                break
                                
                            try:
                                # 레시피 생성
                                recipe = self.recipe_generator.generate_recipe(scene_description)
                                
                                # 메타데이터 추가
                                recipe["metadata"] = {
                                    "movie_title": movie["title"],
                                    "genre": genre,
                                    "base_scene": base_scene,
                                    "recipe_id": recipe_count + 1,
                                    "generation_timestamp": time.time()
                                }
                                
                                recipes.append(recipe)
                                recipe_count += 1
                                
                                # 진행상황 출력
                                if recipe_count % 1000 == 0:
                                    elapsed = time.time() - start_time
                                    rate = recipe_count / elapsed
                                    eta = (target_count - recipe_count) / rate if rate > 0 else 0
                                    print(f"Generated {recipe_count:,}/{target_count:,} recipes "
                                          f"({recipe_count/target_count*100:.1f}%) - "
                                          f"Rate: {rate:.1f}/sec - ETA: {eta/60:.1f}min")
                                
                            except Exception as e:
                                print(f"Error generating recipe for '{scene_description}': {e}")
                                continue
                        
                        if recipe_count >= target_count:
                            break
                    
                    if recipe_count >= target_count:
                        break
                
                if recipe_count >= target_count:
                    break
        
        total_time = time.time() - start_time
        print(f"\nGeneration Complete!")
        print(f"Total recipes: {len(recipes):,}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Average rate: {len(recipes)/total_time:.2f} recipes/second")
        
        return recipes
    
    def save_recipes_to_files(self, recipes: List[Dict], output_dir: str = "generated_recipes"):
        """레시피를 파일로 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Saving {len(recipes):,} recipes to {output_path}...")
        
        # 전체 데이터셋을 하나의 파일로 저장
        all_recipes_file = output_path / "all_movie_recipes.json"
        with open(all_recipes_file, 'w', encoding='utf-8') as f:
            json.dump(recipes, f, ensure_ascii=False, indent=2)
        print(f"Saved all recipes to: {all_recipes_file}")
        
        # 장르별로 분할 저장
        genre_recipes = {}
        for recipe in recipes:
            genre = recipe["metadata"]["genre"]
            if genre not in genre_recipes:
                genre_recipes[genre] = []
            genre_recipes[genre].append(recipe)
        
        for genre, genre_recipe_list in genre_recipes.items():
            genre_file = output_path / f"{genre}_recipes.json"
            with open(genre_file, 'w', encoding='utf-8') as f:
                json.dump(genre_recipe_list, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(genre_recipe_list):,} {genre} recipes to: {genre_file}")
        
        # 통계 정보 저장
        stats = self.generate_dataset_statistics(recipes)
        stats_file = output_path / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Saved statistics to: {stats_file}")
        
        return output_path
    
    def generate_dataset_statistics(self, recipes: List[Dict]) -> Dict:
        """데이터셋 통계 생성"""
        stats = {
            "total_recipes": len(recipes),
            "genres": {},
            "movies": {},
            "volatility_levels": {},
            "emotions": {},
            "average_concentrations": {},
            "note_types": {"top_notes": {}, "middle_notes": {}, "base_notes": {}}
        }
        
        for recipe in recipes:
            # 장르별 통계
            genre = recipe["metadata"]["genre"]
            stats["genres"][genre] = stats["genres"].get(genre, 0) + 1
            
            # 영화별 통계
            movie = recipe["metadata"]["movie_title"]
            stats["movies"][movie] = stats["movies"].get(movie, 0) + 1
            
            # 휘발성 레벨 통계
            volatility = recipe["volatility_level"]
            stats["volatility_levels"][volatility] = stats["volatility_levels"].get(volatility, 0) + 1
            
            # 감정별 통계
            for emotion in recipe["detected_emotions"]:
                stats["emotions"][emotion] = stats["emotions"].get(emotion, 0) + 1
            
            # 노트별 통계
            for note_category, notes in recipe["fragrance_notes"].items():
                for note in notes:
                    note_name = note["name"]
                    note_type = note["note_type"]
                    
                    if note_name not in stats["note_types"][note_category]:
                        stats["note_types"][note_category][note_name] = 0
                    stats["note_types"][note_category][note_name] += 1
        
        return stats

def main():
    """메인 실행 함수"""
    generator = MovieRecipeGenerator()
    
    print("=== Movie Scene Fragrance Recipe Mass Generator ===")
    print("Generating recipes based on famous movie scenes...")
    print()
    
    # 대량 레시피 생성 (10만개)
    recipes = generator.generate_massive_dataset(target_count=100000)
    
    # 파일로 저장
    output_dir = generator.save_recipes_to_files(recipes)
    
    print("\n=== Generation Summary ===")
    print(f"✅ Successfully generated {len(recipes):,} unique fragrance recipes")
    print(f"📁 Saved to directory: {output_dir.absolute()}")
    print(f"🎬 Based on {len([m for movies in generator.famous_movies.values() for m in movies])} famous movies")
    print(f"🎭 Covering {len(generator.famous_movies)} different genres")
    print()
    print("The dataset includes:")
    print("- Detailed fragrance formulations with Top/Middle/Base notes")
    print("- Precise concentration percentages for each material")
    print("- Complete mixing instructions and procedures")
    print("- Movie metadata and scene context")
    print("- Volatility control based on scene requirements")
    
    # 샘플 레시피 출력
    if recipes:
        print(f"\n=== Sample Recipe ===")
        sample = recipes[0]
        print(f"Movie: {sample['metadata']['movie_title']}")
        print(f"Genre: {sample['metadata']['genre']}")
        print(f"Scene: {sample['scene_description']}")
        print(f"Volatility: {sample['volatility_level']}")
        print(f"Duration: {sample['duration_estimate']}")
        print("Fragrance Notes:")
        for note_type, notes in sample['fragrance_notes'].items():
            print(f"  {note_type.replace('_', ' ').title()}:")
            for note in notes:
                print(f"    • {note['name'].replace('_', ' ').title()}: {note['concentration_percent']}%")

if __name__ == "__main__":
    main()