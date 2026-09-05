# built_in_evaluation.py

import opik
import requests
from opik.evaluation import evaluate
from opik.evaluation.metrics import Contains

# Initialize Opik client
client = opik.Opik()

# Create or load dataset
dataset = client.get_or_create_dataset("fine-tuned-model-eval")

# Insert test cases
dataset.insert([
    {
        "input": "What is machine learning?"
    },
    {
        "input": "Explain neural networks"
    }
])


def model_task(dataset_item):
    """Send input to local model API and return response"""
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "message": dataset_item["input"],
                "temperature": 0.1,
                "max_tokens": 200
            },
            timeout=30
        )

        response.raise_for_status()
        return {"output": response.json()["response"]}

    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return {"output": "Error: Model response failed"}


# Built-in keyword checks
scoring_metrics = [
    # Machine learning keywords
    Contains(reference="algorithm"),
    Contains(reference="data"),
    Contains(reference="pattern"),
    Contains(reference="prediction"),

    # Neural network keywords
    Contains(reference="neurons"),
    Contains(reference="layers"),
    Contains(reference="weights"),
    Contains(reference="activation")
]


# Run evaluation
evaluation = evaluate(
    dataset=dataset,
    task=model_task,
    scoring_metrics=scoring_metrics,
    experiment_name="Fine-tuned Model Evaluation"
)

print("✅ Evaluation complete!")
print(f"📊 Results available in Opik dashboard: {evaluation}")
