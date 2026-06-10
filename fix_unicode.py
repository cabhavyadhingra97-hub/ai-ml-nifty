import pathlib, re

for f in ['src/model_training.py', 'src/walk_forward.py']:
    txt = pathlib.Path(f).read_text(encoding='utf-8')
    # Replace n_jobs=-1 with n_jobs=1 to avoid multiprocessing issues on Python 3.14
    txt2 = txt.replace('n_jobs=-1', 'n_jobs=1')
    pathlib.Path(f).write_text(txt2, encoding='utf-8')
    print('Fixed n_jobs in: ' + f)
print('Done.')
