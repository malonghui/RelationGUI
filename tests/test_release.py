import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_relationqa import score
from tools.build_release import canonical,valid_box


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.gold=[{'id':'a','answer':'trigger','platform':'web'}, {'id':'b','answer':'none','platform':'mobile'}]

    def test_perfect(self):
        r=score(self.gold,[{'id':x['id'],'prediction':x['answer']} for x in self.gold])
        self.assertEqual(r['scores']['overall']['accuracy'],1)

    def test_missing_is_wrong(self):
        r=score(self.gold,[{'id':'a','prediction':'trigger'}])
        self.assertEqual(r['scores']['overall']['accuracy'],.5)
        self.assertEqual(r['missing'],1)

    def test_strict_exact_match(self):
        r=score(self.gold,[{'id':'a','prediction':'Trigger'},{'id':'b','prediction':'none.'}])
        self.assertEqual(r['invalid'],2)
        self.assertEqual(r['scores']['overall']['accuracy'],0)

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            score(self.gold,[{'id':'a','prediction':'trigger'}]*2)

    def test_unknown_rejected(self):
        with self.assertRaises(ValueError):
            score(self.gold,[{'id':'c','prediction':'trigger'}])


class SourceTests(unittest.TestCase):
    def test_variant_grouping(self):
        self.assertEqual(canonical('desktop/data_low_macOS/App/view_line_cropped_3.png'),'desktop/data_macOS/App/view.png')
        self.assertEqual(canonical('/root/data/all/mobile/a_low.jpg'),'mobile/a.jpg')

    def test_bounds(self):
        self.assertTrue(valid_box([[0,0],[100,200]],100,200))
        for points in [[[-1,0],[50,20]],[[0,0],[101,20]],[[1,1],[1,5]],[[0,0],[float('nan'),5]]]:
            self.assertFalse(valid_box(points,100,200))


if __name__=='__main__':
    unittest.main()
