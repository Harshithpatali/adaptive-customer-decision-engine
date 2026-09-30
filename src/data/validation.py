import pandas as pd
EXPECTED={"recency","history_segment","history","mens","womens","zip_code","newbie","channel","segment","visit","conversion","spend"}
def load_and_validate(path):
    df=pd.read_csv(path); df.columns=[str(c).strip().lower().replace(' ','_').replace('-','_') for c in df.columns]
    missing=EXPECTED-set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    df['zip_code']=df['zip_code'].replace({'Surburban':'Suburban'})
    # Hillstrom has repeated covariate/outcome patterns; no customer ID exists, so identical rows are not treated as an ingestion error.
    if not df['conversion'].isin([0,1]).all(): raise ValueError('conversion must be binary')
    if not df['visit'].isin([0,1]).all(): raise ValueError('visit must be binary')
    if (df['spend']<0).any(): raise ValueError('spend cannot be negative')
    if df['segment'].nunique()!=3: raise ValueError('Expected three treatment arms')
    return df
