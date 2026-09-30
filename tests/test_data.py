from src.data.validation import load_and_validate
def test_schema():
 d=load_and_validate('data/raw/hillstrom_raw.csv'); assert d.shape==(64000,12); assert set(d.segment.unique())=={'No E-Mail','Womens E-Mail','Mens E-Mail'}
