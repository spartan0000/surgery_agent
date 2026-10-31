import pytest

from agents.calculators import run_calculator, optional_bool, CalculatorInputError

def test_meld_missing_inr():
    result = run_calculator("meld", {"creatinine": 1.8, "bilirubin": 2.1})
    assert "error" in result and "inr" in result["error"]

def test_meld_non_numeric():
    result = run_calculator("meld", {"creatinine": "high", "bilirubin": 2.1, "inr": 1.4})
    assert "error" in result

def test_meld_known_value():
    result = run_calculator("meld", {"creatinine": 1.0, "bilirubin": 1.0, "inr": 1.0})
    assert result["meld"] == 6   # floor value

def test_optional_bool_strings():
    assert optional_bool({"x": "False"}, "x") is False
    assert optional_bool({"x": "yes"}, "x") is True
    with pytest.raises(CalculatorInputError):
        optional_bool({"x": "ture"}, "x")