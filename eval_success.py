# advanced_evaluation.py
import opik
from opik import track
import requests
from opik.evaluation import evaluate
from opik.evaluation.metrics import Contains, Equals, LevenshteinRatio
import json
import uuid
from typing import Any, List
from opik.evaluation.metrics import Hallucination,GEval,AgentTaskCompletionJudge #base_metric, score_result, AgentTaskCompletionJudge
"""
class KeywordChecker(base_metric.BaseMetric):
    def __init__(self, name: str = "keyword_checker"):
        super().__init__(name)

    def score(self, output: str, keywords: List[str], **ignored_kwargs: Any) -> score_result.ScoreResult:
        
        #Check if output contains any of the keywords from metadata.
        
        #Args:
          #  output: The model's output text
          #  keywords: List of keywords from metadata
        
        # Check each keyword individually (case-insensitive)
        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in output.lower():
                found_keywords.append(keyword)
        
        # Calculate score as percentage of keywords found
        score = len(found_keywords) / len(keywords) if keywords else 0.0
        
        return score_result.ScoreResult(
            name=self.name,
            value=score,
            reason=f"Found {len(found_keywords)}/{len(keywords)} keywords: {', '.join(found_keywords)}" if found_keywords else f"No keywords found from: {', '.join(keywords)}"
        )
"""


def load_jsonl_without_ids(file_path):
    """Load JSONL file without adding IDs - let Opik handle it"""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # Remove id field if present - let Opik generate it
                    if "id" in item:
                        del item["id"]
                    
                    # Ensure required fields exist
                    if "input" not in item:
                        print(f"Warning: Line {line_num} missing 'input' field")
                        continue
                        
                    data.append(item)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
        
    return data

def main():
    # Create Opik client
    client = opik.Opik()
    
    # Create datasets for different evaluation sets
    train_dataset = client.get_or_create_dataset("fine-tuned-model-eval-train")
    test_dataset = client.get_or_create_dataset("fine-tuned-model-eval-test") 
    validation_dataset = client.get_or_create_dataset("fine-tuned-model-eval-validation")
    
    # Load data into datasets from JSONL files
    # Note: insert() method is used to add data, not read_jsonl_from_file()
    try:
        # You'll need to load the JSONL data first, then insert it
        # Example of how to load and insert JSONL data:
        import json
        
        # Load train data
        
        train_data=load_jsonl_without_ids("./data/Train Set/dataset_gen_success.jsonl")
        train_dataset.insert(train_data)
        
        # Load test data  
        
        test_data=load_jsonl_without_ids("./data/Test Set/test_orig.jsonl")
        test_dataset.insert(test_data)
        
        # Load validation data

        val_data=load_jsonl_without_ids("./data/Validation Set/val.jsonl")
        validation_dataset.insert(val_data)
        
    except FileNotFoundError as e:
        print(f"Warning: Could not load data files: {e}")
        print("Please ensure your JSONL files exist at the specified paths")
    
    # Initialize evaluation metrics
    levenshtein_metric = LevenshteinRatio()
    

    agent_task = AgentTaskCompletionJudge(model="ollama/glm-5:cloud",
    temperature=0.1,
    track=True,
    project_name="my-evaluation-project-success"
    )

    # Custom graded evaluation
    graded_accuracy = GEval(
        task_introduction="You evaluate the factual accuracy of AI responses on a graduated scale.",
        evaluation_criteria="""
        Score from 0 to 10 based on factual accuracy:
        
        10: Perfect accuracy, all facts correct
        8-9: Mostly accurate with minor errors or omissions
        6-7: Partially accurate, mix of correct and incorrect information
        4-5: More errors than accurate information
        2-3: Mostly inaccurate with few correct elements
        0-1: Completely inaccurate or fabricated
        
        Consider:
        - Factual correctness of claims
        - Completeness of information
        - Presence of unsupported assertions
        - Overall reliability of the response
        """,
        model="ollama/glm-5:cloud",
        temperature=0.1
    )

    payload = """TASK: the research paper by identifying the system purpose, core technologies used, security mechanisms, IoT capabilities, key functionalities, application domains, performance benefits, and research motivation.

    OUTCOME: The agent correctly described the system as an AI-powered virtual voice assistant designed to enhance human-computer interaction and productivity. It identified Python, Google Text-to-Speech (gTTS), AI/ML, NLP, and PyTorch as core technologies. The agent explained that secure face recognition, user authentication, encrypted communication, and privacy safeguards ensure secure access and data protection. It noted IoT integration enables seamless interaction with smart devices. The agent listed functionalities including voice-based web search, application control, reminders, task automation, and data retrieval. It highlighted applications in smart homes, education, healthcare, and productivity environments. The agent also reported improved accuracy, speed, adaptability, and efficiency compared to traditional assistants. Finally, it explained the research motivation to overcome limitations of current voice assistants, improve contextual understanding, and develop intelligent, human-centric systems.
    """

    score = agent_task.score(output=payload)
    hallucination_metric = Hallucination(
    model="ollama/glm-5:cloud",  # Use your Ollama model
    temperature=0.1
    )

    contains_metric = Contains(case_sensitive=False)
    equals_metric = Equals()
    # Quick metric test
    # Contains will be True because output contains the expected text
    test_output = "The review highlights specific security vulnerabilities that IoT devices face when interconnected with other systems."
    test_expected = "security vulnerabilities"

    print("🧪 Quick metric test:")
    print(f"Contains result: {contains_metric.score(output=test_output, reference=test_expected)}")  # 1.0
    print(f"Equals result: {equals_metric.score(output=test_output, reference=test_expected)}")      # 0.0

    @track
    def model_task(dataset_item):
        """Task function for evaluation with proper error handling"""
        try:
            response = requests.post(
                "http://localhost:8000/v1/chat/completions",
                json={
                    "model": "qwen3-4b-lora-adapter",
                    "messages": [
                        {"role": "user", "content": dataset_item["input"]}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200
                },
                timeout=60,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            output = result["choices"][0]["message"]["content"]
            
            return {
                "input": dataset_item["input"],
                "output": output
            }
            
        except Exception as e:
            print(f"Error in model_task: {e}")
            return {
                "input": dataset_item["input"],
                "output": f"Error: {str(e)}"
            }
    
    # Run evaluations on different datasets
    datasets_to_evaluate = [
        (train_dataset, "train"),
        (test_dataset, "test"), 
        (validation_dataset, "validation")
    ]
    
    for dataset, dataset_type in datasets_to_evaluate:
        print(f"🔄 Running evaluation on {dataset_type} dataset...")
        
        try:
            evaluation = evaluate(
                dataset=dataset,
                task=model_task,
                scoring_metrics=[graded_accuracy], #contains_metric, levenshtein_metric, equals_metric, agent_task,hallucination_metric
                experiment_name=f"fine-tuned-model-evaluation-{dataset_type}",
                # Optional: Add key mapping if your dataset fields don't match metric expectations
                scoring_key_mapping={
                    "reference": "expected_output",  # Map dataset field to metric input
                }
            )
            
            print(f"✅ {dataset_type.capitalize()} evaluation complete!")
            print(f"   Experiment ID: {evaluation.experimentId}")
            print(f"   Total test cases: {len(evaluation.testResults)}")
            
        except Exception as e:
            print(f"❌ Error during {dataset_type} evaluation: {e}")
    
    print("📊 All evaluations complete! Check Opik dashboard for results")

if __name__ == "__main__":
    main()
