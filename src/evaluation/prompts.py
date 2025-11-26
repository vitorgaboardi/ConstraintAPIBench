NATURALNESS_EVALUATION = """Naturalness refers to how realistic an utterance is compared to how a real user would typically request a service.

Evaluation Criteria:
- NATURAL: The utterance reads like a genuine user query. It uses conversational language, may include casual phrasing, and sounds like something a person would actually say when interacting with a chatbot.
    Examples: 
    - "Find me cheap flights to New York"
    - "Show me available hotels" 
    - "I need a restaurant near me"

- UNNATURAL: The utterance sounds artificial, overly technical, or like generated/robotic text. It may include unusual phrasing, awkward parameter names, or API-like language.
    Examples: 
    - "Retrieve all aviations offerings from new york destination with minimal cost parameters"
    - "fetch humidity for lat=32.918, long=12.112 using OpenWeather"
    - "Get me the information about the movie with ID iAdcaj976jx23"

Guidelines:
1. Focus on whether the phrasing/structure sounds like human speech
2. Technical terms are acceptable if used naturally (e.g., "filter by price" is natural; "apply filtering on cost attribute" is not)

Respond only with 'natural' or 'unnatural'. Choose the one that best fits."""