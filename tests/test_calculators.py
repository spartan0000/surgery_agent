import pytest

from agents.calculators import run_calculator

def test_meld_missing_inr():
    result = run_calculator("meld", {"creatinine": 1.8, "bilirubin": 2.1})
    assert "error" in result and "inr" in result["error"]

def test_meld_non_numeric():
    result = run_calculator("meld", {"creatinine": "high", "bilirubin": 2.1, "inr": 1.4})
    assert "error" in result

def test_meld_known_value():
    result = run_calculator("meld", {"creatinine": 1.0, "bilirubin": 1.0, "inr": 1.0})
    assert result["meld"] == 6   # floor value