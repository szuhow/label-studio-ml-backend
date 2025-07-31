#!/usr/bin/env python3
"""
Skrypt testowy dla multi-model backend
"""

import requests
import json
import sys
import argparse

def test_multi_model_backend(base_url="http://localhost:9090"):
    """Testuje działanie multi-model backend"""
    
    print(f"🔍 Testowanie Multi-Model Backend: {base_url}")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Status: {health_data.get('status')}")
            print(f"   Modele załadowane: {health_data.get('models_loaded')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Lista modeli
    print("\n2. Lista dostępnych modeli...")
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            models_data = response.json()
            print(f"✅ Znaleziono {models_data.get('total_models', 0)} modeli:")
            
            for model_name, model_info in models_data.get('available_models', {}).items():
                endpoint = model_info.get('endpoint', '/')
                model_type = model_info.get('model_type', 'unknown')
                resolution = model_info.get('resolution', '?')
                status = model_info.get('status', 'unknown')
                default_marker = " (DEFAULT)" if model_name == models_data.get('default_model') else ""
                
                print(f"   • {model_name}{default_marker}")
                print(f"     Endpoint: {endpoint}")
                print(f"     Typ: {model_type}")
                print(f"     Rozdzielczość: {resolution}px")
                print(f"     Status: {status}")
                print()
                
            return models_data
        else:
            print(f"❌ Lista modeli failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Lista modeli error: {e}")
        return False

def test_prediction_endpoint(base_url, endpoint="/", model_name="default"):
    """Test konkretnego endpointu predykcji"""
    
    print(f"\n3. Test predykcji - {model_name} ({endpoint})...")
    
    # Minimalna konfiguracja zadania (bez rzeczywistego obrazu)
    test_task = {
        "tasks": [
            {
                "data": {
                    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wICh1c2luZyBJSkcgSlBFRyB2NjIpLCBxdWFsaXR5ID0gOTAK/9sAQwADAgIDAgIDAwMDBAMDBAUIBQUEBAUKBwcGCAwKDAwLCgsLDQ4SEA0OEQ4LCxAWEBETFBUVFQwPFxgWFBgSFBUU/9sAQwEDBAQFBAUJBQUJFA0LDRQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQU/8AAEQgAAQABAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBkQgUQqGxwdEVUvAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD5/ooooA/9k="
                }
            }
        ]
    }
    
    try:
        url = f"{base_url}{endpoint}predict" if endpoint != "/" else f"{base_url}/predict"
        response = requests.post(
            url,
            json=test_task,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Predykcja udana")
            print(f"   Status: {response.status_code}")
            
            if 'results' in result and len(result['results']) > 0:
                first_result = result['results'][0]
                model_version = first_result.get('model_version', 'unknown')
                score = first_result.get('score', 0)
                print(f"   Model version: {model_version}")
                print(f"   Score: {score}")
                
                result_count = len(first_result.get('result', []))
                print(f"   Wyniki segmentacji: {result_count}")
                
            return True
        else:
            print(f"❌ Predykcja failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Predykcja error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Multi-Model Backend')
    parser.add_argument('--url', default='http://localhost:9090', 
                       help='Base URL of the backend')
    parser.add_argument('--test-prediction', action='store_true',
                       help='Test prediction endpoints (requires real backend)')
    
    args = parser.parse_args()
    
    # Test podstawowych funkcji
    models_data = test_multi_model_backend(args.url)
    
    if not models_data:
        print("\n❌ Podstawowe testy nie powiodły się!")
        sys.exit(1)
    
    # Test predykcji (opcjonalny)
    if args.test_prediction and isinstance(models_data, dict):
        print("\n" + "=" * 60)
        print("TESTY PREDYKCJI")
        
        # Test domyślnego modelu
        test_prediction_endpoint(args.url, "/", "default")
        
        # Test wszystkich innych modeli
        for model_name, model_info in models_data.get('available_models', {}).items():
            endpoint = model_info.get('endpoint', '/')
            if endpoint != "/":  # Skip default model (already tested)
                test_prediction_endpoint(args.url, endpoint, model_name)
    
    print("\n" + "=" * 60)
    print("✅ TESTY ZAKOŃCZONE")

if __name__ == "__main__":
    main()
