import json

instructions = []
with open('train_best.json', 'r') as f:
    data = json.load(f)
    for key, chunk in data.items():
        for pairs in chunk['generated']:
            question, answer = pairs['question'], pairs['answer']
            context_pair = {
                'question': f"{pairs['question']}",
                'answer': pairs['answer']
            }
            instructions.append(context_pair)

with open('data/instruction.json', 'w') as f:
    json.dump(instructions, f)

with open('data/instruction.json', 'r') as f:
    data = json.load(f)
