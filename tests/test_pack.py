import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.build_release import pack


class PackageTests(unittest.TestCase):
    def test_final_archives_are_complete_and_checksums_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('RelationGUI','RelationQA'):
                (root/name).mkdir()
                (root/name/'README.md').write_text('test payload',encoding='utf-8')
            pack(root)
            self.assertFalse(list(root.glob('*.partial')))
            for line in (root/'SHA256SUMS').read_text().splitlines():
                checksum,name=line.split('  ',1)
                archive=root/name
                self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),checksum)
                with zipfile.ZipFile(archive) as z:
                    self.assertIsNone(z.testzip())
                    self.assertEqual(len(z.namelist()),1)


if __name__=='__main__':
    unittest.main()
