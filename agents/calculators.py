#calculators 

import math
import json

# can't remember all the places to update each tool so here's a list
# 
# 1. Implement calculate_X(params: dict) -> dict
# 2. Add "x": calculate_X to run_calculator dict
# 3. Update @beta_tool medical_calculator docstring in agent.py
# 4. Update tool_definitions medical_calculator in agent.py
#    - add to enum list
#    - add description with parameters

class CalculatorInputError(Exception):
    """Raised when calculator inputs are missing or invalid."""
    pass

def require_number(params, key, min_val=None, max_val=None):
    if key not in params or params[key] is None:
        raise CalculatorInputError(f"Missing required parameter: {key}")
    try:
        value = float(params[key])
    except (TypeError, ValueError):
        raise CalculatorInputError(f"{key} must be a number, got {params[key]!r}")
    if min_val is not None and value < min_val:
        raise CalculatorInputError(f"{key}={value} is below plausible minimum {min_val}")
    if max_val is not None and value > max_val:
        raise CalculatorInputError(f"{key}={value} is above plausible maximum {max_val}")
    return value

def require_choice(params, key, choices):
    if key not in params:
        raise CalculatorInputError(f"Missing required parameter: {key}")
    value = str(params[key]).lower()
    if value not in choices:
        raise CalculatorInputError(f"{key} must be one of {choices}, got {value!r}")
    return value

def optional_bool(params, key, default=False):
    return bool(params.get(key, default))



def calculate_meld(params: dict) -> dict:
    """
    MELD score = 3.78 x ln(bilirubin) + 
                 11.2 x ln(INR) + 
                 9.57 x ln(creatinine) + 
                 6.43
    MELD-Na adds sodium correction
    """
    creatinine = min(require_number(params, "creatinine", 0.1, 30), 4.0)  # capped at 4.0
    bilirubin = max(require_number(params, "bilirubin", 0.1, 80), 1.0)    # minimum of 1.0
    inr = max(require_number(params, "inr", 0.5, 15), 1.0)
    
    # If on dialysis twice in past week, creatinine = 4.0
    if optional_bool(params, "dialysis"):
        creatinine = 4.0
    
    meld = (3.78 * math.log(bilirubin) + 
            11.2 * math.log(inr) + 
            9.57 * math.log(creatinine) + 
            6.43)
    
    meld = min(math.floor(max(meld, 6) + 0.5), 40)  # minimum score of 6 and maximum score of 40
    
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
    
    if params.get("sodium") is not None:
        sodium = max(min(require_number(params, "sodium", 90, 180), 137), 125) #sodium bounded between 125 and 137
        meld_na = meld + 1.32 * (137 - sodium) - (0.033 * meld * (137 - sodium))
        result["meld_na"] = math.floor(meld_na + 0.5)
    
    else:
        result["meld"] = math.floor(meld + 0.5)
    return result


def calculate_bmi(params: dict) -> dict:
    height_unit = require_choice({"height_unit": params.get("height_unit") or "cm"},
                                 "height_unit", ("cm", "inches"))
    weight_unit = require_choice({"weight_unit": params.get("weight_unit") or "kg"},
                                 "weight_unit", ("kg", "pounds"))

    # Plausible ranges depend on the unit given
    if height_unit == "inches":
        height_cm = require_number(params, "height", 20, 100) * 2.54
    else:
        height_cm = require_number(params, "height", 50, 250)
    if weight_unit == "pounds":
        weight_kg = require_number(params, "weight", 4, 1500) * 0.453592
    else:
        weight_kg = require_number(params, "weight", 2, 700)
    
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



def calculate_chads_vasc(params: dict) -> dict:
    """
    CHA2DS2-VASc score for stroke risk in atrial fibrillation
    Lip et al. 2010
    """
    score = 0
    criteria_met = []
    
    female = optional_bool(params, "female")

    # Congestive heart failure
    if optional_bool(params, "heart_failure"):
        score += 1
        criteria_met.append("heart failure")
    
    # Hypertension
    if optional_bool(params, "hypertension"):
        score += 1
        criteria_met.append("hypertension")
    
    # Age ≥ 75 (2 points)
    age = int(require_number(params, "age", 0, 120))
    if age >= 75:
        score += 2
        criteria_met.append("age ≥ 75")
    elif age >= 65:
        score += 1
        criteria_met.append("age 65-74")
    
    # Diabetes
    if optional_bool(params, "diabetes"):
        score += 1
        criteria_met.append("diabetes")
    
    # Stroke/TIA/thromboembolism (2 points)
    if optional_bool(params, "stroke_tia"):
        score += 2
        criteria_met.append("prior stroke/TIA")
    
    # Vascular disease (prior MI, PAD, aortic plaque)
    if optional_bool(params, "vascular_disease"):
        score += 1
        criteria_met.append("vascular disease")
    
    # Sex category female
    if female:
        score += 1
        criteria_met.append("female sex")
    
    # Annual stroke risk interpretation
    stroke_risk = {
        0: "0.2%",
        1: "0.6%",
        2: "2.2%",
        3: "3.2%",
        4: "4.8%",
        5: "7.2%",
        6: "9.7%",
        7: "11.2%",
        8: "10.8%",
        9: "12.2%"
    }

    stroke_tia_systemic_embolism_risk = {
        0: "0.3%",
        1: "0.9%",
        2: "2.9%",
        3: "4.6%",
        4: "6.7%",
        5: "10.0%",
        6: "13.6%",
        7: "15.7%",
        8: "15.2%",
        9: "17.4%"
    }
    
    # Anticoagulation recommendation
    # Note: female sex alone (score=1) doesn't warrant anticoagulation
    male_score = score - (1 if female else 0)
    
    if male_score == 0:
        anticoagulation = "Not recommended"
        bridging = "No bridging needed for surgery"
        surgical_risk = "Low — anticoagulation can be stopped"
    elif male_score == 1:
        anticoagulation = "Consider anticoagulation"
        bridging = "Bridging decision based on bleeding vs thrombotic risk"
        surgical_risk = "Low-moderate — discuss with cardiologist"
    else:
        anticoagulation = "Anticoagulation recommended"
        if score >= 4:
            bridging = "Bridging anticoagulation likely needed"
            surgical_risk = "High — cardiology input recommended before stopping anticoagulation"
        else:
            bridging = "Consider bridging based on procedure bleeding risk"
            surgical_risk = "Moderate — weigh thrombotic vs bleeding risk"
    
    return {
        "score": score,
        "annual_stroke_risk": stroke_risk.get(score, ">15%"),
        "stroke_TIA_systemic_embolism_risk": stroke_tia_systemic_embolism_risk.get(score, ">17%"),
        "criteria_met": criteria_met,
        "anticoagulation_recommendation": anticoagulation,
        "perioperative_bridging": bridging,
        "surgical_context": surgical_risk,
        "note": "Clinical decision requires assessment of procedural bleeding risk alongside thrombotic risk. Consult cardiology for high risk cases."
    }

###
#function to actually call the calculator functions themselves

def run_calculator(name: str, params: dict) -> dict:
    
    #Dict mapping string names to function objects
    
    calculators = {
        "meld": calculate_meld,
        "bmi": calculate_bmi,
        #"child_pugh": calculate_child_pugh,
        "chads_vasc": calculate_chads_vasc,
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
    except CalculatorInputError as e:
        return {"error": str(e), "calculator": name, "action": "ask_user_for_correct_value"}
    except KeyError as e:
        return {"error": f"Missing required parameter: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "calculator": name}