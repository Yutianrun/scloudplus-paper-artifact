"""Check full-size accounting and reject incomplete or mismatched captures."""
import csv
from pathlib import Path
import tempfile
import unittest

from run import capture_stats


class AccountingTests(unittest.TestCase):
    def fixture(self,path,n,nbar,alpha='.01',duplicate=False,bad_calls=False):
        fields=['coord','position','n_a','n_b','guess','ref_ok']
        counts=[0]*6
        with (path/'capture.log').open('w') as f:
            f.write('# probe_validation=cached-public-buckets-v1\n')
            writer=csv.writer(f);writer.writerow(fields)
            for i in range(n*nbar):
                c,p=divmod(i,n)
                if duplicate and i==1:c,p=0,0
                writer.writerow([c,p,32,48,0,1]);counts[i%6]+=1
        text=''.join(f'CAL worker={w} attempt=1 status=ok\n' for w in range(6))
        for w,count in enumerate(counts):
            calls=6400+224*count+int(bad_calls and w==0)
            text+=f'SPRT_DONE worker={w} coefficients={count} calls={calls} seconds=10.0\n'
        text+=f'SPRT_FINISHED coefficients={n*nbar} undone=0 cap=1000 alpha={alpha} workers=6 wall=10.0\n'
        (path/'capture.err').write_text(text)

    def test_full_384_and_512_shapes_and_calls(self):
        for n in [1664,2400]:
            with self.subTest(n=n),tempfile.TemporaryDirectory() as d:
                p=Path(d);self.fixture(p,n,16)
                r=capture_stats(p,n,16)
                self.assertEqual(r['coefficients'],n*16)
                self.assertEqual(r['mean_samples_per_axis'],40)
                self.assertEqual(r['timing_calls'],38400+224*n*16)

    def test_reject_wrong_alpha_duplicate_or_unaccounted_calls(self):
        for kwargs in [dict(alpha='.001'),dict(duplicate=True),dict(bad_calls=True)]:
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as d:
                p=Path(d);self.fixture(p,4,2,**kwargs)
                with self.assertRaises(ValueError):capture_stats(p,4,2)


if __name__=='__main__':unittest.main()
