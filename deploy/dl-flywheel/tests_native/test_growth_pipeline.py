import sqlite3, unittest
from growth_pipeline import reserve,queue_counts,STAGE,check_new_row
class QueueTests(unittest.TestCase):
 def setUp(self):
  self.db=sqlite3.connect(':memory:',isolation_level=None)
  self.db.executescript('CREATE TABLE variants(sequence INTEGER PRIMARY KEY,parent TEXT,ordinal INTEGER,state TEXT,identity TEXT UNIQUE,created REAL,UNIQUE(parent,ordinal));CREATE TABLE '+STAGE+'(identity TEXT,stage TEXT,receipt_json TEXT,receipt_sha TEXT,created REAL,PRIMARY KEY(identity,stage));')
  self.parents=[{'parent_lineage_sha256':'p'+str(i)}for i in range(300)]
 def tearDown(self):self.db.close()
 def add(self,n,state='published'):
  for i in range(n):self.db.execute('INSERT INTO variants VALUES(?,?,?,?,?,0)',(i,'p'+str(i),0,state,'i'+str(i)))
 def test_new_publication_rows_above_old_limit(self):
  row={'state':'generated','sequence':138};check_new_row(row,dict(row))
  with self.assertRaises(ValueError):check_new_row(row,{'state':'published','sequence':138})
  with self.assertRaises(ValueError):check_new_row({'state':'generated','sequence':127},{'state':'generated','sequence':127})
 def test_published128_does_not_wait_for_training(self):
  self.add(128);r=reserve(self.db,self.parents,100000,128,4096);self.assertEqual(r[0],128);self.assertEqual(queue_counts(self.db)['admitted'],0)
 def test_raw_bound(self):
  self.add(128,'generated');self.assertIsNone(reserve(self.db,self.parents,100000,128,4096))
 def test_reserved_unknown_occupies_raw(self):
  self.add(128,'reserved');self.assertIsNone(reserve(self.db,self.parents,100000,128,4096))
 def test_encoding_bound(self):
  self.add(128);self.assertIsNone(reserve(self.db,self.parents,100000,128,128))
 def test_encoding_releases_only_encoder_bound(self):
  self.add(128);self.db.execute('INSERT INTO '+STAGE+" VALUES('i0','kumo_small_s3_verified','{}','a',0)")
  self.assertIsNotNone(reserve(self.db,self.parents,100000,128,128));self.assertEqual(queue_counts(self.db)['admitted'],0)
 def test_wrong_stage_or_unknown_identity_no_capacity(self):
  self.add(128);self.db.execute('INSERT INTO '+STAGE+" VALUES('i0','inferred','{}','a',0)");self.db.execute('INSERT INTO '+STAGE+" VALUES('unknown','kumo_small_s3_verified','{}','a',0)")
  self.assertIsNone(reserve(self.db,self.parents,100000,128,128))
 def test_target(self):
  self.add(128);self.assertIsNone(reserve(self.db,self.parents,128,128,4096))
 def test_parent_round_robin(self):
  self.add(300);r=reserve(self.db,self.parents,100000,128,4096);self.assertEqual(r,(300,self.parents[0],1))
 def test_insert_failure_rolls_back(self):
  self.add(1);self.parents[1]=self.parents[0]
  with self.assertRaises(sqlite3.IntegrityError):reserve(self.db,self.parents,100000,128,4096)
  self.assertFalse(self.db.in_transaction);self.assertEqual(self.db.execute('SELECT count(*) FROM variants').fetchone()[0],1)
if __name__=='__main__':unittest.main()
