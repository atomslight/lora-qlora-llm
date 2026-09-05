def prompt_base(data: str, num_records: int = 15) -> str:
    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
the provided data chunk into diverse question and answer (Q&A) pairs that will be used to fine-tune a language model.
For each of the {num_records} entries, generate well-structured questions that reflect different aspects of the information in the chunk.
 
Ensure the following diversity in your questions:
1. **Question Types:** Include a mix of:
   - Factual questions (direct facts from the text)
   - Inferential questions (require reasoning from the text)
   - Comparative questions (compare two concepts or ideas)
   - True/False questions (boolean answer with explanation)
 
2. **Difficulty Levels:** Include a mix of:
   - Easy (1-2 sentence answer)
   - Medium (2-3 sentence answer)
   - Hard (3-4 sentence answer requiring deeper understanding)
 
3. **Question Length:** Mix of:
   - Short questions (1-2 sentences)
   - Long questions (3-4 sentences with context)
 
4. **Strict Rules:**
   - No repetitive or similar questions
   - No sensitive or biased content
   - Answers must be accurate, neutral and grounded in the provided data
   - Each question must target a different aspect of the data
   - Do not copy sentences directly, rephrase naturally
 
Structure your output in JSON format, where each object contains 'question' and 'answer' fields. The JSON structure should look like this:
    "question": "Your question here...",
    "answer": "Your answer here..."
 
Example:
    "question": "What is the primary purpose of this dataset?",
    "answer": "This dataset serves as training data for fine-tuning a language model to improve its performance on domain specific tasks."
 
    "question": "Is it true that fine-tuning always requires a large dataset? Explain.",
    "answer": "False, fine-tuning can be done with smaller datasets if the data is high quality and diverse enough to cover the domain."
 
    "question": "Considering the relationship between data quality and model performance, how does the diversity of training examples influence the ability of a fine-tuned model to generalize to unseen data?",
    "answer": "Diverse training examples expose the model to varied patterns and contexts, enabling it to generalize better to unseen data rather than memorizing specific patterns."
 
By following these guidelines, you will contribute to a robust and effective dataset that enhances the model performance.
 
Data
{data}
"""

def prompt_temp1(data: str, num_records: int = 15) -> str:
    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
the provided data chunk into diverse question and answer (Q&A) pairs that will be used to fine-tune a language model to prevent hallucinations.
For each of the {num_records} entries, generate well-structured questions that reflect different aspects of the information in the chunk.
 
Ensure the following diversity in your questions:
1. **Question Types:** Include a mix of:
   - False Premise questions (include a plausible but factually incorrect assumption)
   - Misconception questions (ask about a common misunderstanding related to the text)
   - Out-of-Scope questions (ask about something slightly outside the provided text)
 
2. **Difficulty Levels:** Include a mix of:
   - Easy (1-2 sentence answer correcting a simple premise)
   - Medium (2-3 sentence answer explaining why the assumption is wrong)
   - Hard (3-4 sentence answer detailing the nuance of the correction)
 
3. **Question Length:** Mix of:
   - Short questions (1-2 sentences)
   - Long questions (3-4 sentences with context setting up the false premise)
 
4. **Strict Rules:**
   - No repetitive or similar questions
   - No sensitive or biased content
   - For False Premise questions, the answer must politely but firmly correct the user using exact facts from the text
   - For Out-of-Scope questions, the answer MUST explicitly state that the information is not provided, before summarizing what IS available
   - Answers must be 100% grounded in the provided data; no external knowledge
   - Do not copy sentences directly, rephrase naturally
 
Structure your output in JSON format, where each object contains 'question' and 'answer' fields. The JSON structure should look like this:
    "question": "Your question here...",
    "answer": "Your answer here..."
 
Example:
    "question": "Since fine-tuning completely replaces a model's base knowledge, how much data do I need to rewrite its entire memory?",
    "answer": "Actually, fine-tuning does not completely replace a model's base knowledge; it is primarily used to adjust the model's behavior and performance on domain-specific tasks. You can achieve this with smaller, high-quality datasets rather than needing enough data to rewrite its entire memory."
 
By following these guidelines, you will contribute to a robust and effective dataset that enhances the model performance.
 
Data
{data}
"""

def prompt_temp2(data: str, num_records: int = 15) -> str:
    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
the provided data chunk into diverse question and answer (Q&A) pairs that will be used to fine-tune a language model for complex reasoning.
For each of the {num_records} entries, generate well-structured questions that reflect different aspects of the information in the chunk.
 
Ensure the following diversity in your questions:
1. **Question Types:** Include a mix of:
   - Multi-Hop questions (require connecting at least two separate facts from different parts of the text)
   - Synthesis questions (require combining details to form a new conclusion stated in the text)
   - Relational questions (ask how one concept in the text directly affects another)
 
2. **Difficulty Levels:** Include a mix of:
   - Medium (2-3 sentence answer connecting two facts)
   - Hard (3-4 sentence answer explicitly walking through the logical steps to reach the conclusion)
 
3. **Question Length:** Mix of:
   - Short questions (1-2 sentences)
   - Long questions (3-4 sentences with context)
 
4. **Strict Rules:**
   - No repetitive or similar questions
   - No simple, direct-lookup factual questions
   - No sensitive or biased content
   - Answers must explicitly walk through the logical steps or connections made to reach the conclusion
   - Answers must be accurate, neutral and grounded in the provided data
   - Do not copy sentences directly, rephrase naturally
 
Structure your output in JSON format, where each object contains 'question' and 'answer' fields. The JSON structure should look like this:
    "question": "Your question here...",
    "answer": "Your answer here..."
 
Example:
    "question": "Based on the relationship between data quality and model generalization, what would likely happen if we used a massive but highly repetitive dataset for fine-tuning?",
    "answer": "If a massive but highly repetitive dataset is used, the model would likely memorize specific patterns rather than generalizing well to unseen data. This is because the text indicates that diverse training examples, not just volume, are what expose the model to varied contexts required for effective generalization."
 
By following these guidelines, you will contribute to a robust and effective dataset that enhances the model performance.
 
Data
{data}
"""

def prompt_temp3(data: str, num_records: int = 15) -> str:
    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
the provided data chunk into diverse question and answer (Q&A) pairs that will be used to fine-tune a language model to act as a helpful assistant.
For each of the {num_records} entries, generate well-structured questions that reflect different aspects of the information in the chunk.
 
Ensure the following diversity in your questions:
1. **Question Types:** Include a mix of:
   - Troubleshooting scenarios (user describes a problem related to the text)
   - Application scenarios (user asks how to apply a concept from the text to their specific project)
   - Advisory scenarios (user asks for recommendations based on the text's facts)
 
2. **Difficulty Levels:** Include a mix of:
   - Easy (1-2 sentence answer providing a direct solution)
   - Medium (2-3 sentence answer explaining how to apply the concept)
   - Hard (3-4 sentence answer providing a multi-step recommendation)
 
3. **Question Length:** Mix of:
   - Short questions (1-2 sentences)
   - Long questions (3-4 sentences outlining the user's specific fictional scenario)
 
4. **Strict Rules:**
   - No repetitive or similar questions
   - Frame all questions in the first-person ("I am trying to...", "My team is...")
   - Answers must adopt the persona of a helpful, professional assistant
   - Answers must solve the scenario using ONLY facts grounded in the provided data
   - Do not copy sentences directly, rephrase naturally
 
Structure your output in JSON format, where each object contains 'question' and 'answer' fields. The JSON structure should look like this:
    "question": "Your question here...",
    "answer": "Your answer here..."
 
Example:
    "question": "I'm working on a tight budget and only have a few hundred examples of customer support chats. Can I still fine-tune a model for my startup, or do I need to scrape thousands more?",
    "answer": "Yes, you can absolutely still fine-tune your model. As long as your few hundred examples are of high quality and diverse enough to cover the specific domain of your customer support tasks, smaller datasets can be highly effective for fine-tuning."
 
By following these guidelines, you will contribute to a robust and effective dataset that enhances the model performance.
 
Data
{data}
"""

def prompt_temp4(data: str, num_records: int = 15) -> str:
    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
the provided data chunk into diverse question and answer (Q&A) pairs that will be used to fine-tune a language model to adapt its tone.
For each of the {num_records} entries, generate well-structured questions that reflect different aspects of the information in the chunk.
 
Ensure the following diversity in your questions:
1. **Question Types (Audience):** Include a mix of:
   - Beginner/ELI5 questions (asked by a novice needing simple analogies)
   - Intermediate questions (standard inquiries)
   - Expert questions (highly technical or analytical inquiries)
 
2. **Difficulty Levels (Tone):** Include a mix of:
   - Easy (1-2 sentence answer using simple vocabulary and analogies)
   - Medium (2-3 sentence standard, professional answer)
   - Hard (3-4 sentence answer that is dense, precise, and highly technical)
 
3. **Question Length:** Mix of:
   - Short questions (1-2 sentences)
   - Long questions (3-4 sentences with context)
 
4. **Strict Rules:**
   - No repetitive or similar questions
   - The answer tone MUST perfectly match the expertise level implied by the question
   - Beginner answers must simplify the text's concepts without losing accuracy
   - Expert answers must synthesize the most complex parts of the text
   - Answers must be accurate, neutral and grounded in the provided data
   - Do not copy sentences directly, rephrase naturally
 
Structure your output in JSON format, where each object contains 'question' and 'answer' fields. The JSON structure should look like this:
    "question": "Your question here...",
    "answer": "Your answer here..."
 
Example:
    "question": "I am totally new to AI. What exactly does 'fine-tuning' mean in simple terms?",
    "answer": "Think of the base AI model like a general college graduate who knows a little bit about everything. Fine-tuning is like sending them to a specific training bootcamp—by showing them specialized examples, you improve their performance on a very specific job or domain."
 
By following these guidelines, you will contribute to a robust and effective dataset that enhances the model performance.
 
Data
{data}
"""

def prompt_temp5(data: str, num_records: int = 15) -> str:
    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
the provided data chunk into diverse question and answer (Q&A) pairs that will be used to train a language model's latent reasoning.
For each of the {num_records} entries, generate well-structured questions that reflect different aspects of the information in the chunk.
 
Ensure the following diversity in your questions:
1. **Question Types:** Include a mix of:
   - "How" questions (uncovering mechanisms described in the text)
   - "Why" questions (explaining the reasoning behind a fact in the text)
   - Evaluation questions (assessing a concept based on the text's rules)
 
2. **Difficulty Levels:** Include a mix of:
   - Medium (2-3 sentence answer)
   - Hard (3-4 sentence answer requiring deeper step-by-step understanding)
 
3. **Question Length:** Mix of:
   - Short questions (1-2 sentences)
   - Long questions (3-4 sentences with context)
 
4. **Strict Rules:**
   - No repetitive or similar questions
   - No simple "What is" lookup questions
   - EVERY answer MUST begin with "Thinking Process: [1-2 sentences explaining how the answer is deduced from the text]. Final Answer: [The actual factual answer]."
   - Answers must be accurate, neutral and grounded in the provided data
   - Do not copy sentences directly, rephrase naturally
 
Structure your output in JSON format, where each object contains 'question' and 'answer' fields. The JSON structure should look like this:
    "question": "Your question here...",
    "answer": "Your answer here..."
 
Example:
    "question": "Why is it beneficial to ensure high diversity in the training examples when preparing a dataset?",
    "answer": "Thinking Process: The text states that diverse examples expose the model to varied patterns and contexts. This exposure directly enables the model to generalize better to unseen data rather than just memorizing things. Final Answer: High diversity is beneficial because it exposes the model to a wide variety of patterns and contexts, which prevents the model from simply memorizing specific inputs and improves its ability to generalize to new, unseen data."
 
By following these guidelines, you will contribute to a robust and effective dataset that enhances the model performance.
 
Data
{data}
"""

def prompt_temp6(data: str, num_records: int = 15) -> str:
    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
the provided data chunk into diverse question and answer (Q&A) pairs that will be used to fine-tune a language model for strict instruction adherence.
For each of the {num_records} entries, generate well-structured questions that reflect different aspects of the information in the chunk.
 
Ensure the following diversity in your questions:
1. **Question Types (Constraints):** Include a mix of:
   - Bullet point constraints ("List the features in 3 bullet points")
   - Word limit constraints ("Explain this in under 20 words")
   - Format constraints ("Provide the answer as a comma-separated list")
   - Structured comparison constraints ("Compare A and B using a structured format")
 
2. **Difficulty Levels:** Include a mix of:
   - Easy (Simple list or short constraint)
   - Medium (Structured explanation constraint)
   - Hard (Synthesizing complex info into strict formatting constraints)
 
3. **Question Length:** Mix of:
   - Short questions (1-2 sentences)
   - Long questions (3-4 sentences with context)
 
4. **Strict Rules:**
   - No repetitive or similar questions
   - Every question MUST explicitly request a specific formatting constraint
   - The answer MUST flawlessly execute the requested formatting constraint
   - Answers must be accurate, neutral and grounded in the provided data
   - Do not copy sentences directly, rephrase naturally
 
Structure your output in JSON format, where each object contains 'question' and 'answer' fields. The JSON structure should look like this:
    "question": "Your question here...",
    "answer": "Your answer here..."
 
Example:
    "question": "Summarize the key requirements for a fine-tuning dataset in exactly three bullet points.",
    "answer": "- The data must be of high quality.\n- The training examples must be diverse enough to cover the target domain.\n- The dataset must expose the model to varied patterns to encourage generalization."
 
By following these guidelines, you will contribute to a robust and effective dataset that enhances the model performance.
 
Data
{data}
"""

# Execution block to test the functions
if __name__ == "__main__":
    # Sample data to test
    sample_data = "Nicholas Renotte is an AI Engineer and YouTuber who creates tutorials on Data Science, Machine Learning, and Large Language Models."
    
    # We call prompt_temp1 as an example to show it working
    #generated_text = prompt_temp1(data=sample_data, num_records=10)
    
    print("--- GENERATED PROMPT PREVIEW ---")
    print(sample_data)
