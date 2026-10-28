#agent tool definitions
tool_definitions = [
    {
        "name": "search_surgical_textbook",
        "description": """Search the surgical textbook for operative
techniques, anatomy, procedures and complications.
Use for procedural and anatomical questions.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_cochrane_citations",
        "description": """Search Cochrane systematic reviews for
high quality synthesized clinical evidence.
Use for questions about treatment effectiveness,
outcomes and comparative evidence.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_bmi",
        "description": "Calculate BMI from height and weight",
        "input_schema": {
            "type": "object",
            "properties": {
                "height": {"type": "number"},
                "weight": {"type": "number"},
                "height_unit": {
                    "type": "string",
                    "enum": ["cm", "inches"]
                },
                "weight_unit": {
                    "type": "string",
                    "enum": ["kg", "pounds"]
                }
            },
            "required": ["height", "weight"]
        }
    }
]