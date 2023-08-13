import sys
sys.path.append("..")
import unittest
import pandas as pd
import data_analysis

class TestLoadData(unittest.TestCase):

    def test_load_data(self):
        existing_df = pd.read_csv('existing_data.csv')
        expected_result = pd.read_csv('expected_result.csv')
        result = data_analysis.load_additional_data(existing_df, 'new_data.csv')
        self.assertDictEqual(result.to_dict(orient='list'), expected_result.to_dict(orient='list'))

if __name__ == '__main__':
    unittest.main()