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
        "name": "medical_calculator",
        "description": """Calculate medical risk scores and measurements.
        
Available calculators:

bmi: Body mass index
    height (number), weight (number)
    height_unit (cm or inches, default cm)
    weight_unit (kg or pounds, default kg)

meld: MELD score for liver disease severity
    creatinine (mg/dL), bilirubin (mg/dL), inr
    sodium (optional), dialysis (bool, optional)

chads_vasc: CHA2DS2-VASc stroke risk in atrial fibrillation
    Useful for perioperative anticoagulation decisions
    heart_failure (bool), hypertension (bool),
    age (years), diabetes (bool), stroke_tia (bool),
    vascular_disease (bool), female (bool)

rcri: Revised Cardiac Risk Index for surgical cardiac risk
    high_risk_surgery (bool), ischemic_heart_disease (bool),
    heart_failure (bool), cerebrovascular_disease (bool),
    insulin_dependent_diabetes (bool), creatinine_gt_177 (bool)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "calculator": {
                    "type": "string",
                    "description": "Calculator to use",
                    "enum": ["bmi", "meld", "chads_vasc", "rcri"]
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for the calculator"
                }
            },
            "required": ["calculator", "parameters"]
        }
    }
]