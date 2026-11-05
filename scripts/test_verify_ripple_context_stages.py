"""CPU-only regression checks for exact context-budget fitting."""
import unittest
from verify_ripple_context_stages import FitError, fit_message, legacy_fit_message, ids_sha256


class FitTests(unittest.TestCase):
    def test_newline_merge_gap_reduces_one_unit(self):
        builder=lambda repeat,padding:(repeat,padding)
        encode=lambda pair:list(range(pair[0]*24+(pair[1]+1 if pair[1] else 0)))
        with self.assertRaises(FitError) as caught:
            legacy_fit_message(241,builder,encode,24)
        self.assertEqual([x['delta'] for x in caught.exception.trace[1:]],[1,-1]*4)
        content,ids,_=fit_message(241,builder,encode,24)
        self.assertEqual(content,(9,24))
        self.assertEqual(len(ids),241)

    def test_repeat_estimate_must_be_actually_bracketed(self):
        content,ids,fit=fit_message(10000,lambda n,p:(n,p),lambda pair:list(range(sum(pair))),24)
        self.assertEqual(len(ids),10000)
        self.assertTrue(any(x.get('repeat',0)>1000 for x in fit['trace']))

    def test_legacy_success_is_reproducible(self):
        builder=lambda n,p:(n,p)
        encode=lambda pair:list(range(pair[0]*24+pair[1]))
        first=legacy_fit_message(245,builder,encode,24)
        second=legacy_fit_message(245,builder,encode,24)
        self.assertEqual(first,second)
        self.assertEqual(len(first[1]),245)
        self.assertEqual(ids_sha256(first[1]),ids_sha256(second[1]))


if __name__=='__main__':unittest.main()
