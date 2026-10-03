"""Participant leakage protections at the manifest and partition boundaries."""
import csv
import tempfile
import unittest
from pathlib import Path
from fidi.manifest import fold_partition, read_manifest

class ManifestCases(unittest.TestCase):
    def write_manifest(self,root,rows):
        for row in rows:
            (root/row[1]).write_bytes(b"placeholder")
        p=root/"manifest.csv"
        with p.open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["image_id","image_path","mask_path","participant_id","recording_id","split","fold"])
            w.writerows(rows)
        return p

    def test_valid_patient_grouped_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[["a","a.png","a.png","p1","r1","development","1"],
                  ["b","b.png","b.png","p2","r2","development","2"],
                  ["c","c.png","c.png","p3","r3","test",""],
                  ["d","d.png","d.png","p1","r4","development","1"]]
            parsed=read_manifest(self.write_manifest(root,rows))
            train,val,test=fold_partition(parsed,1)
            self.assertEqual((len(train),len(val),len(test)),(1,2,1))
            self.assertFalse({r["participant_id"] for r in train} & {r["participant_id"] for r in val})

    def test_cross_fold_and_test_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[["a","a.png","a.png","p1","r1","development","1"],
                  ["b","b.png","b.png","p1","r2","development","2"]]
            p=self.write_manifest(root,rows)
            with self.assertRaisesRegex(ValueError,"multiple development folds"):read_manifest(p)
            rows[1][-2:]=["test",""]
            p=self.write_manifest(root,rows)
            with self.assertRaisesRegex(ValueError,"development and test"):read_manifest(p)
            with self.assertRaisesRegex(ValueError,"cannot be disabled"):read_manifest(p,group_folds_by_participant=False)

    def test_partition_defends_against_unvalidated_rows(self):
        rows=[{"participant_id":"p1","split":"development","fold":"1"},
              {"participant_id":"p1","split":"development","fold":"2"}]
        with self.assertRaisesRegex(ValueError,"training and validation"):fold_partition(rows,1)
        rows[1]["participant_id"]="p2"
        rows.append({"participant_id":"p1","split":"test","fold":""})
        with self.assertRaisesRegex(ValueError,"development and test"):fold_partition(rows,1)

if __name__=="__main__":unittest.main()
