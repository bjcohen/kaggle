import pandas as pd

i = 1

def df_to_libsvm(df, target, filename, columns = None, colmap = None):
    if columns is None:
        columns = list(df.columns)
    if target in columns:
        columns.remove(target)
    if colmap is None:
        max_index = 0
        colmap = {}
        for c in columns:
            colmap.update({c + '_' + str(name) : index for (index, name) in enumerate(set(df[c]), max_index)})
            max_index += len(set(df[c]))
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            f.write(str(row[target]))
            f.write(' ')
            features = [colmap[col + '_' + str(value)] for col, value in row.iteritems() if col != target]
            f.write(' '.join(['%d:1'% col for col in sorted(features)]))
            f.write('\n')
    return colmap
