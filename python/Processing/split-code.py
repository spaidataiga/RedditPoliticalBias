 #!/usr/bin/env python
import pandas as pd
import glob
import os

for f in glob.glob("selfdata/processed/2019-*.csv"):
    df = pd.read_csv(f, chunksize=10000)
    base = f[19:-4]
    if not os.path.exists(f'selfdata/processed/splits/{base}'):
        os.makedirs(f'selfdata/processed/splits/{base}')
    for i,chunk in enumerate(df):
        chunk.to_csv(f"selfdata/processed/splits/{base}/{base}_{i:04d}.csv", index=False)
    del df
