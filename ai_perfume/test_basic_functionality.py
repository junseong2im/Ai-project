#!/usr/bin/env python3
"""
Basic functionality test for the AI perfume system
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from models.text_analyzer import TextAnalyzer
        from models.recipe_generator import RecipeGenerator
        from core.learning_system import LearningSystem
        from core.database import Session, Recipe, Feedback, init_db
        from config.settings import Config
        print("[OK] All imports successful")
        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False

def test_text_analysis():
    """Test text analysis functionality"""
    try:
        from models.text_analyzer import TextAnalyzer
        analyzer = TextAnalyzer()
        
        test_text = "여름 저녁 해변의 시원한 바람"
        result = analyzer.analyze(test_text)
        
        print("[OK] Text analysis successful")
        print(f"  - Emotions: {list(result['emotions'].keys())}")
        print(f"  - Keywords: {result['keywords'][:3]}...")
        print(f"  - Embeddings shape: {result['embeddings'].shape}")
        return True
    except Exception as e:
        print(f"[ERROR] Text analysis error: {e}")
        return False

def test_recipe_generation():
    """Test recipe generation functionality"""
    try:
        from models.text_analyzer import TextAnalyzer
        from models.recipe_generator import RecipeGenerator
        import torch
        
        analyzer = TextAnalyzer()
        generator = RecipeGenerator()
        
        test_text = "여름 저녁 해변의 시원한 바람"
        analysis = analyzer.analyze(test_text)
        
        emotion_scores = torch.tensor(
            [list(analysis['emotions'].values())],
            dtype=torch.float32
        )
        
        recipe = generator.generate_recipe(
            analysis['embeddings'],
            emotion_scores
        )
        
        print("[OK] Recipe generation successful")
        print(f"  - Top notes: {recipe['top_notes']}")
        print(f"  - Middle notes: {recipe['middle_notes']}")
        print(f"  - Base notes: {recipe['base_notes']}")
        print(f"  - Intensity: {recipe['intensity']:.1f}/10")
        return True
    except Exception as e:
        print(f"[ERROR] Recipe generation error: {e}")
        return False

def test_database():
    """Test database functionality"""
    try:
        from core.database import init_db, get_db_session, Recipe
        
        # Initialize database
        init_db()
        
        # Test session
        with get_db_session() as session:
            # Query test
            count = session.query(Recipe).count()
            print("[OK] Database test successful")
            print(f"  - Recipe count: {count}")
        return True
    except Exception as e:
        print(f"[ERROR] Database error: {e}")
        return False

def main():
    """Run all tests"""
    print("Running AI Perfume System Tests...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Text Analysis Test", test_text_analysis),
        ("Recipe Generation Test", test_recipe_generation),
        ("Database Test", test_database),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! The system is working correctly.")
    else:
        print("[WARNING] Some tests failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)