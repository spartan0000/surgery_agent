#calculators 

import math
import json

def calculate_meld(params: dict) -> dict:
    """
    MELD score = 3.78 x ln(bilirubin) + 
                 11.2 x ln(INR) + 
                 9.57 x ln(creatinine) + 
                 6.43
    MELD-Na adds sodium correction
    """
    creatinine = min(float(params["creatinine"]), 4.0)  # capped at 4.0
    bilirubin = max(float(params["bilirubin"]), 1.0)    # minimum of 1.0
    inr = max(float(params["inr"]), 1.0)
    
    # If on dialysis twice in past week, creatinine = 4.0
    if params.get("dialysis"):
        creatinine = 4.0
    
    import math
    meld = (3.78 * math.log(bilirubin) + 
            11.2 * math.log(inr) + 
            9.57 * math.log(creatinine) + 
            6.43)
    
    meld = min(round(max(meld, 6), 1), 40)  # minimum score of 6 and maximum score of 40
    
    # 90-day mortality interpretation
    if meld < 10:
        mortality = "< 1.9%"
        urgency = "Low"
    elif meld < 20:
        mortality = "6%"
        urgency = "Moderate"
    elif meld < 30:
        mortality = "19.6%"
        urgency = "High"
    elif meld < 40:
        mortality = "52.6%"
        urgency = "Very High"
    else:
        mortality = "> 71.3+%"
        urgency = "Critical"
    
    # MELD-Na if sodium provided
    result = {
        "meld": meld,
        "90_day_mortality": mortality,
        "urgency": urgency
    }
    
    if "sodium" in params:
        sodium = max(min(float(params["sodium"]), 137), 125) #sodium bounded between 125 and 137
        meld_na = meld + 1.32 * (137 - sodium) - (0.033 * meld * (137 - sodium))
        result["meld_na"] = round(meld_na, 1)
    
    return result


def calculate_bmi(params: dict) -> dict:
    try:
        height_cm = float(params["height"])
        weight_kg = float(params["weight"])
    except (KeyError, ValueError) as e:
        return {"error":f"Invalid input: {e}"}

    if height_cm <= 0 or weight_kg <= 0:
        return {"error":"Height and weight must be positive"}
    
    # Unit conversion
    if params.get("height_unit") == "inches":
        height_cm = height_cm * 2.54
    if params.get("weight_unit") == "pounds":
        weight_kg = weight_kg * 0.453592
    
    bmi = weight_kg / (height_cm / 100) ** 2
    
    category = (
        "underweight" if bmi < 18.5 else
        "normal" if bmi < 25 else
        "overweight" if bmi < 30 else
        "obese class I" if bmi < 35 else
        "obese class II" if bmi < 40 else
        "obese class III"
    )
    
    return {
        "bmi": round(bmi, 1),
        "category": category,
        "height_used_cm": round(height_cm, 1),
        "weight_used_kg": round(weight_kg, 1)
    }


###
#function to actually call the calculator functions themselves

def run_calculator(name: str, params: dict) -> dict:
    
    #Dict mapping string names to function objects
    
    calculators = {
        "meld": calculate_meld,
        #"child_pugh": calculate_child_pugh,
        #"chads_vasc": calculate_chads_vasc,
        #"rcri": calculate_rcri
    }
    
    # Check if requested calculator exists
    if name not in calculators:
        return {
            "error": f"Unknown calculator: {name}",
            "available": list(calculators.keys())
        }
    
    # Call the function and return result
    try:
        return calculators[name](params)
    except KeyError as e:
        return {"error": f"Missing required parameter: {e}"}
    except Exception as e:
        return {"error": str(e)}