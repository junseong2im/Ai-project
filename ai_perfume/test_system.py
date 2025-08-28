#!/usr/bin/env python3
"""
Movie Scent AI System Test
Simple test without unicode characters for Windows console
"""

import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

def test_system():
    """Test the movie scent AI system"""
    print("="*50)
    print("Movie Scent AI System - Test Mode")
    print("="*50)
    
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Test imports
    try:
        print("Testing core imports...")
        from core.optimized_data_manager import OptimizedDataManager
        from core.movie_scent_ai import AdvancedMovieNeuralNetwork
        print("* Core imports successful")
        
        # Test data manager
        print("Testing data manager...")
        data_manager = OptimizedDataManager()
        scenes = data_manager.search_scenes_by_emotion("love")
        print(f"* Found {len(scenes)} love scenes")
        
        # Test web server
        print("Starting web server...")
        import uvicorn
        
        def run_server():
            uvicorn.run(
                "app:app",
                host="127.0.0.1", 
                port=8000,
                log_level="error"
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        time.sleep(3)
        print("* Web server started at http://localhost:8000")
        
        # Open browser
        webbrowser.open("http://localhost:8000")
        print("* Browser opened")
        
        print("\nSystem Status: READY")
        print("Web Interface: http://localhost:8000")
        print("API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nSystem stopped.")
            
    except Exception as e:
        print(f"* Error: {e}")
        return False
        
    return True

if __name__ == "__main__":
    test_system()