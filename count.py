import json
import re

pattern = re.compile(r'\bquestion\b')

def count_in_text(text):
    return len(pattern.findall(text.lower()))

def process_obj(obj):
    total = 0
    
    if isinstance(obj, dict):
        combined_text = " ".join(str(v) for v in obj.values())
        total += count_in_text(combined_text)
    
    elif isinstance(obj, list):
        for item in obj:
            total += process_obj(item)
    
    return total

total = 0
count_objects = 0

with open("instruction.json", "r", encoding="utf-8") as f:
    content = f.read()

decoder = json.JSONDecoder()
idx = 0
length = len(content)

while idx < length:
    content = content[idx:].lstrip()
    if not content:
        break
    
    try:
        obj, offset = decoder.raw_decode(content)
        
        total += process_obj(obj)
        count_objects += 1
        
        idx += offset
    except json.JSONDecodeError:
        break

print("Total occurrences:", total)
print("Chunks parsed:", count_objects)
