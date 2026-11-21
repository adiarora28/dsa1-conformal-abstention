from dsa1.grading import clean_generation, exact_match, grade, normalize_answer, token_f1


def test_normalize():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"


def test_clean_generation():
    assert clean_generation("Answer: Paris.\nBecause...") == "Paris"
    assert clean_generation("<think></think>\n\n\"BRCA1\"") == "BRCA1"


def test_exact_match_and_f1():
    assert exact_match("the Beatles", ["Beatles", "The Fab Four"])
    assert not exact_match("", ["x"])
    assert token_f1("Barack Obama", ["Obama"]) == 2 / 3
    g = grade("Barack Obama", ["Obama"])
    assert g["correct_em"] is False and g["correct_f1"] is True
