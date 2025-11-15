PROMPT_UTTERANCE_GENERATION = """
Imagine that you are a user who wants to utilize the features provided by various APIs in your daily life. Your task is to come up with realistic scenarios for using these APIs and express them as natural language instructions, as if you were asking a friend or assistant for help.

Please follow these guidelines:
1. The instructions should be 1 to 2 sentences long. Use a mix of interrogative sentences, first-person statements, imperative sentences, and other structures that convey a request. Aim for diversity in your instructions.
2. Do not mention the API's name in your instructions.
3. Your instructions should only involve the features provided by these APIs. The instructions that need multiple times of API call is better.
4. Generate 10 diverse instructions.
5. Use specific nouns and real-world examples from various domains, such as entertainment, sports, or technology. Avoid using any form of placeholder or generic phrases, such as "this xxx", "a xxx" or "a specific xxx", and provide concrete details instead.
6. Try not to repeat the verb for each instruction to maximize diversity.
7. Ensure diversity in language by combining questions with imperative statements and other structures that convey a request.
    
The output must be a Python list of dictionaries, where each dictionary has two keys:
    - "utterance": the natural language request.
    - "parameters": a dictionary containing the name-value pairs for all parameters used in the utterance. 
    
The "parameters" dictionary must have pairs that can be infered or recognised from the generated natural utterance. 
Finally, you must only output the Python list and do not output anything else, such as notes or explanation about the reasoning.
"""