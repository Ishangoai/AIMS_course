from utils.utility_functions import multiply_by_two


def test_multiply_by_two():
    actual_output = multiply_by_two(4)
    expected_output = 8
    assert actual_output == expected_output
