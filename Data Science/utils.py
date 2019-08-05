import pandas as pd

def get_map(groupby, column):
    df = data.groupby(groupby)[column].value_counts()
    print(df.head(10))
    keys = list(data[groupby].unique())
    
    mapping = data[groupby].map({
        key: df[key].idxmax()  if key in df.index else np.nan for key in keys
    })
    return mapping
