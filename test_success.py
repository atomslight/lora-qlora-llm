# test_integration.py
import requests
import json
import time

def test_model_with_opik():
    """Test your model and check if traces appear in Opik"""
    
    print("🧪 Testing model integration with Opik...")
    
    # Test your model API
    response = requests.post(
        "http://localhost:8000/chat",
        json={
            "message": "What is artificial intelligence?",
            "temperature": 0.1,
            "max_tokens": 150
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Model response: {result['response'][:100]}...")
        
        # Wait a moment for trace to be logged
        time.sleep(2)
        
        print("📊 Check your Opik dashboard at http://localhost:5173")
        print("   You should see a new trace in the 'fine-tuned-llm-api' project")
        
    else:
        print(f"❌ Model API error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_model_with_opik()
# test_integration.py
import requests
import json
import time

def test_model_with_opik():
    """Test your model and check if traces appear in Opik"""
    
    print("🧪 Testing model integration with Opik...")
    
    # Test your model API
    response = requests.post(
        "http://localhost:8000/chat",
        json={
            "message": "What is artificial intelligence?",
            "temperature": 0.1,
            "max_tokens": 150
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Model response: {result['response'][:100]}...")
        
        # Wait a moment for trace to be logged
        time.sleep(2)
        
        print("📊 Check your Opik dashboard at http://localhost:5173")
        print("   You should see a new trace in the 'fine-tuned-llm-api' project")
        
    else:
        print(f"❌ Model API error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_model_with_opik()
