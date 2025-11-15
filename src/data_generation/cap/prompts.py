# constraint extraction: 
CONSTRAINT_EXTRACTION = """You are an expert in interpreting OpenAPI specification (OAS). Your task is to extract constraints for each parameter of the API method.
    I will provide you with the API name and description, the API method name and description, and the list of parameters, each with its name and description. 
    For each parameter, you must extract constraints by analysing the OAS provided. Do not create constraints if they are not defined in the documentation.
    Extract the following constraints:

(1) format: Any required input format, such as general or standard codes (e.g., ISO 8601 date/time, currency codes, country codes, IATA codes). Do not infer formats based on default examples.
(2) values: Constraints about the values parameters can assume. This includes: 
    - "max": for maximum numeric value, 
    - "min": for minimum numeric value, 
    - "enumerated": for a closed set of only acceptable values. Only use "enumerated" if a specific closed list of possible values is allowed (e.g., ["google", "microsoft", "bing"], ["image", "video", "audio"]). Limit the content to a maximum of 40 values.
(3) id: This constraint must be "True" if the parameter represents a resource identifier (ID) (e.g., parameter names like "message_id", "gameId", "hotelIds", "locationInternalIDs" or descriptions mentioning the parameter represents an ID).
(4) technical: This constraint must be "True" if the parameter is primarily used by developers rather than end users for creating API calls or page formatting. Examples include pagination controls (e.g., page number, offset, page size), authentication tokens (e.g., api key, access token), system management fields (e.g., cache, debug, locale, encoding), and callback URLs. Parameters designed for user interactions and affect sorting or limiting the output (e.g., limit, sort, sortBy, orderBy, etc.) must NOT be "True".
(5) inter-dependency: Describe a required dependency between parameters. The examples mentioned are illustrative for you to understand better. Only include the constraints if stated in the description of the API method or parameter. Dependencies include: 
    - the presence of one parameter requires the presence of another parameter (e.g., "startDate, endDate: if startDate is provided, endDate must also be included.").
    - given a set of parameters, one or more of them must be included in the API call (e.g., "email, phone: at least one of these contact methods must be provided."), 
    - given a set of parameters, one and only of them must be included in the API call (e.g., "q, name and name_equals: only one of them must be used."), 
    - given a set of parameters, either all of them are provided or none of them (e.g., "latitude and longitude: both parameters must be included together."),
    - given a set of parameters, zero or one parameter can be present in the API call (e.g., "promoCode, discountId: at most one of these can be included in a request."), 
    - given a set of parameters, they are related by means of arithmetic constraints (e.g., "minPrice, maxPrice: minPrice must be less than or equal to maxPrice.").

Additional guidelines:
    * For the inter-dependency constraint, list all related parameters separated by commas, followed by a textual description of the constraint.
    * If a parameter includes values constraints ("minimum", "maximum", "enumerated"), you must include them under "values" key.
    * Do not create or use any constraint types that are not listed above (e.g., "require", "mandatory", "optional").
    * If any field is not applicable or cannot be inferred for a parameter, omit that field entirely from the output. 

The final output must be a JSON object where each key is the name of a parameter, and the value is a nested object with only the relevant constraint fields. 
You must strictly follow the output format and do not return anything else other than the JSON object.
"""

EXAMPLE_INPUT_CONSTRAINT = """ 
{
  "API Name": "Flight Search API",
  "API Description": "The Flight Search API retrieves available flights between two locations using IATA codes. It supports one-way and round-trip searches, passenger details, travel class, and paginated results ordered by departure time.",
  "API Method Name": "flightSearch",
  "API Method Description": "Returns a list of flights.",
  "Parameters": [
    {
      "name": "originLocationCode",
      "description": "city/airport [IATA code] from which the traveler will depart, e.g. BOS for Boston",
      "required": true,
    },
    {
      "name": "destinationLocationCode",
      "description": "city/airport [IATA code] to which the traveler is going, e.g. PAR for Paris",
      "required": true,
    },
    {
      "name": "departureDate",
      "description": "the date on which the traveler will depart from the origin to go to the destination. Dates are specified in the [ISO 8601] YYYY-MM-DD format, e.g. 2017-12-25",
      "required": true,
    },
    {
      "name": "returnDate",
      "description": "the date on which the traveler will depart from the destination to return to the origin. If this parameter is not specified, only one-way itineraries are found. If this parameter is specified, only round-trip itineraries are found. Dates are specified in the [ISO 8601] YYYY-MM-DD format, e.g. 2018-02-28". returnDate must not be before the departureDate.
      "required": false,
    },
    {
      "name": "adults",
      "description": "the number of adult travelers (age 12 or older on date of departure). The total number of seated travelers (adult and children) can not exceed 9.",
      "required": true,
    },
    {
      "name": "children",
      "description": "the number of child travelers (older than age 2 and younger than age 12 on date of departure) who will each have their own separate seat. If specified, this number should be greater than or equal to 0. The total number of seated travelers (adult and children) can not exceed 9.",
      "required": false,
    },
    {
      "name": "travelClass",
      "description": "Most of the flight time should be spent in a cabin of this quality or higher. The accepted travel class is economy, premium economy, business or first class. If no travel class is specified, the search considers any travel class. Available classes are: ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST,
      "required": false,
    },
    {
      "name": "page[offset]",
      "description": "start index of the requested page",
      "required": false,
      "default": 0
    },
    {
      "name": "flightId",
      "description": "A unique identifier for the flight. This can be used to retrieve specific flight details.",
      "required": false,
      "example": "FL123456"
    },
    {
      "name": "limit",
      "description": "Number of instances to return. Possible values varies from 1 up to 500.",
      "required": false
    }
  ]
}
"""

EXAMPLE_OUTPUT_CONSTRAINT = """
{
  "originLocationCode": {
    "format": "IATA code"
  },
  "destinationLocationCode": {
    "format": "IATA code"
  },
  "departureDate": {
    "format": "ISO 8601 YYYY-MM-DD format",
  },
  "returnDate": {
    "format": "ISO 8601 YYYY-MM-DD format",
    "inter-dependency": "departureDate, returnDate: returnDate must be equal or after departureDate.
  },
  "adults": {
    "values": {
      "max": 9,
      "min": 1,
    },
    "inter-dependency": "adults, children: The combined number of adults and children must not exceed 9."
  },
  "children": {
    "values": {
      "max": 9,
      "min": 1
    },
    "inter-dependency": "adults, children: The combined number of adults and children must not exceed 9."
  },
  "travelClass": {
    "values": {
      "enumerated": ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]
    }
  },api_related
  "page[offset]": {
    "technical": true
  }
  "flightId": {
    "id": true
  },
  "limit": {
    "values": {
      "max": 500,
      "min": 1
  }
}
"""

PROMPT_UTTERANCE_GENERATION = """You are an expert in interpreting OpenAPI Specification (OAS) and generating utterances and API calls related to an OAS. 
I will provide you with API name and description, API method name and description, and list of parameters, each with its name, description, and constraints.
Your task is to generate utterances and API calls that users may ask that must be solved using the given API method. Consider the following guidelines:

    1) Generated utterances must respect the constraints imposed by the OAS and defined under the "constraints" key over each parameter. This includes the following rules:
        - Required parameters: All required parameters must be included in all utterances and API calls.
        - Value constraints: Values must conform to range limitations (minimum and maximum) and respect the enumerated set of values that the parameter can assume.
        - Format constraints: The parameter values in the API call must strictly conform with the format constraint defined in the OAS.
        - Inter-dependency constraints: They describes constraints between parameters. The combination of parameters values must strictly respect these constraints. For instance, there are cases where two parameters must be added simultaneously in an API call or only one parameter must be included between a group of parameters.

    2) Generated utterances must be semantically related to the API method and closely resemble what users would normally say when interacting with chatbots or requesting for tasks. This includes the following rules:
        - Do not add the API name or the API method name in the utterance. Exceptions may be used when the API is related to a known brand or product (e.g., Spotify for music, IMDb for movies, Booking.com for hotels, Google for search).
        - Do not use general placeholders such as "this URL", "this link", "this name", "a candidate" for parameters. Instead, add the parameter infomration, (e.g., URL, link, name) in the utterance instead of only in the API call. 
        - Every parameter value used in the API call must be explicitly inferable from the utterance. For instance, a link, name, destination mapped in the arguments must be mentioned in the utterance.
        - Do not include parameter values in the utterance with a format that the user would not normally say in the utterance. For instance, instead of using precise latitutude and longitude to represent a place, a more natural utterance would reference a city name or landmark. In other words, the "format" constraint must be strictly respect in the API call but not in the utterance.
        - If the "id" constraint is True, do not generate artificial IDs for those parameters (e.g., "1234" or "abc_123"). Instead, create values that represent what the ID refers to. For example, for "hotelId" parameter, generate hotel names such as "The Grand Hyatt" instead of "ACPAR419". Exceptions are parameters where the user actually have access or knowledge about the ID, such as a tracking package ID, flight number, invoice ticket.

    3) Generated utterances must have a diverse combination of optional parameters. This includes the following rules:
        - Parameter coverage: Include all optional parameters at least once in the generated utterances. The only exception is when the number of optional parameters is too large for the number of utterances requested, which would lead to unnatural utterances.
        - Parameter combination diversity: Use different combinations of optional parameters across utterances to reflect the many ways utterances should be related to the given API.
        - Enumerated values: If a parameter includes an "enumerated" constraint (i.e., a fixed set of allowed values), cycle through different allowed values across the utterances to maximize parameter value diversity.
        - Linguistic diversity: Ensure that utterances have diversity in terms of words (rich vocabulary with different verbs and nouns) and sentence structure (declarative, interrogative, imperative, compound, complex sentence, and so on).
        - Parameter value diversity: Avoid repeating the same value for any parameter across different utterances. The "default" key is just an example and must not be used in all utterances. 
         
The output must be a Python list of dictionaries, where each dictionary has two keys:
    - "utterance": the natural language request.
    - "parameters": a dictionary containing the name-value pairs for all parameters used in the utterance. 

Generate utterances even with APIs that have no parameters. In this case, the "parameters" dictionary must be empty.    
The "parameters" dictionary must have pairs that can be infered or recognised from the generated natural utterance and following the rules defined in the OAS. 
Finally, you must only output the Python list and do not output anything else, such as notes or explanation about the reasoning.
"""