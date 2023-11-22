# test_my_functions.py
import unittest

class TestAddFunction(unittest.TestCase):
    def test_add(self):
        self.assertEqual(2+3, 5)
        self.assertEqual(1+(-1), 0)

if __name__ == '__main__':
    unittest.main()