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



def calculate_chads_vasc(params: dict) -> dict:
    """
    CHA2DS2-VASc score for stroke risk in atrial fibrillation
    Lip et al. 2010
    """
    score = 0
    criteria_met = []
    
    # Congestive heart failure
    if params.get("heart_failure", False):
        score += 1
        criteria_met.append("heart failure")
    
    # Hypertension
    if params.get("hypertension", False):
        score += 1
        criteria_met.append("hypertension")
    
    # Age ≥ 75 (2 points)
    age = int(params.get("age", 0))
    if age >= 75:
        score += 2
        criteria_met.append("age ≥ 75")
    elif age >= 65:
        score += 1
        criteria_met.append("age 65-74")
    
    # Diabetes
    if params.get("diabetes", False):
        score += 1
        criteria_met.append("diabetes")
    
    # Stroke/TIA/thromboembolism (2 points)
    if params.get("stroke_tia", False):
        score += 2
        criteria_met.append("prior stroke/TIA")
    
    # Vascular disease (prior MI, PAD, aortic plaque)
    if params.get("vascular_disease", False):
        score += 1
        criteria_met.append("vascular disease")
    
    # Sex category female
    if params.get("female", False):
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
    male_score = score - (1 if params.get("female", False) else 0)
    
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