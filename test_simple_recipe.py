import sys
sys.path.append('ai_perfume')

from core.scene_fragrance_recipe import SceneFragranceRecipe

# Initialize recipe generator
generator = SceneFragranceRecipe()

print("Movie Scene Fragrance Recipe Generator")
print("=" * 50)

# Test different volatility scenes
test_scenes = [
    "액션 영화의 폭발 장면에서 강렬한 임팩트가 필요한 순간",  # High volatility
    "조용한 도서관에서 혼자 책을 읽는 평화로운 순간",  # Low volatility  
    "로맨틱한 해변 석양에서 커플이 키스하는 장면"  # Medium volatility
]

for i, scene in enumerate(test_scenes, 1):
    print(f"\n[Test {i}] {scene}")
    
    recipe = generator.generate_recipe(scene)
    
    print(f"Volatility Level: {recipe['volatility_level']}")
    print(f"Detected Emotions: {recipe['detected_emotions']}")
    print(f"Duration: {recipe['duration_estimate']}")
    print(f"Diffusion: {recipe['diffusion_range']}")
    
    print("\nFRAGRANCE MATERIALS:")
    for j, material in enumerate(recipe['materials'], 1):
        if material['function'] != 'carrier_solvent':
            print(f"  {j}. {material['name'].replace('_', ' ').title()}")
            print(f"     Concentration: {material['concentration_percent']}%")
            print(f"     Function: {material['function']}")
            print(f"     Volatility: {material['volatility']}%")
    
    # Show carrier
    carrier = next(m for m in recipe['materials'] if m['function'] == 'carrier_solvent')
    print(f"\n  Base: {carrier['name']} {carrier['concentration_percent']}%")
    print("-" * 40)

print("\nKey Features:")
print("- Only recipe and concentration control")
print("- No manufacturing specifications") 
print("- Volatility control based on scene type")
print("- ALL durations kept LOW (seconds to minutes)")
print("- Concentration automatically adjusted for quick dissipation")