import unittest
import utils
from allennlp.data.fields import TextField
from allennlp.data import Token

class TestUtils(unittest.TestCase):
    def test_validate_word_swap(self):
        text_field = TextField([Token("You"), Token("are"), Token("great")])
        modification = {2: "great"}
        result = utils.validate_word_swap(text_field, modification)
        self.assertEqual(result, True)

if __name__ == '__main__':
    unittest.main()