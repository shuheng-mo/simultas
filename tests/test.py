import pytest

# unittest sample
# class TestSomething(unittest.TestCase):
#     def test_sum(self):
#         self.assertEqual(sum([1,2,3]),6, 'equals 6')

#     def test_sum_tuple(self):
#         self.assertEqual(sum((1,2,3)),6,'equals 6')

# if __name__ == '__main__':
#     unittest.main()


# not recommended
# @pytest.fixture
# def expected_result():
#     """pytest fixture example

#     Returns:
#         _type_: anything you will need in a test
#     """
#     return 6


# parameterize the test input
@pytest.mark.parametrize("test_lists, expected_result", [
    ([3, 2, 1], 6),
    ([3, 2], 5),
    ([1, 2, 1], 4),
    ([2, 1], 3),
    ([2], 2),
    ([1], 1),
])
def test_list(test_lists, expected_result):
    assert sum(test_lists) == expected_result, 'Equals'


@pytest.mark.parametrize('test_tuples, expected_result', [
    ((1, 2, 3), 6),
    ((2, 3), 5),
    ((1, 3), 4),
    ((1, 2), 3),
    ((2, ), 2),
    ((1, ), 1),
])
def test_tuple(test_tuples, expected_result):
    assert sum(test_tuples) == expected_result, 'Equals'
