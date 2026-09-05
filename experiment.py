# experiment.py
import opik
from opik import track
import requests

# Initialize Opik client
client = opik.Opik()


@track(project_name="model-evaluation")
def evaluate_model_responses():
    """Evaluate your model on different types of questions"""

    test_questions = [
        "Explain machine learning in simple terms",
        "What are the benefits of neural networks?",
        "How does fine-tuning work?",
        "What is the difference between AI and ML?",
        "Describe transformer architecture"
    ]

    results = []

    for question in test_questions:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "message": question,
                "temperature": 0.1,
                "max_tokens": 200
            }
        )

        if response.status_code == 200:
            answer = response.json()['response']
            results.append({
                "question": question,
                "answer": answer,
                "length": len(answer)
            })

            print(f"Q: {question}")
            print(f"A: {answer[:100]}...\n")

    return results


if __name__ == "__main__":
    results = evaluate_model_responses()
    print(f"✅ Evaluated {len(results)} questions")
    print("📊 Check Opik dashboard for detailed traces")
