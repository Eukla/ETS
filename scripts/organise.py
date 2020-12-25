from os import listdir
from os.path import isfile, join
import pandas as pd

files = \
    ['interesting/' + f for f in listdir('interesting') if isfile(join('interesting', f))]
not_interesting_files = \
    ['not_interesting/' + f for f in listdir('not_interesting') if isfile(join('not_interesting', f))]

files.extend(not_interesting_files)

alive = pd.DataFrame()
apoptotic = pd.DataFrame()
necrotic = pd.DataFrame()
mts = pd.DataFrame()

for file in files:

    c = 0 if 'not_interesting' in file else 1
    df = pd.read_csv(file)
    alive = alive.append(pd.Series([c]).append(df['Alive'], ignore_index=True), ignore_index=True)
    apoptotic = apoptotic.append(pd.Series([c]).append(df['Apoptotic'], ignore_index=True), ignore_index=True)
    necrotic = necrotic.append(pd.Series([c]).append(df['Necrotic'], ignore_index=True), ignore_index=True)

    dft = pd.concat(
        [pd.DataFrame([c, c, c], index=['Alive', 'Apoptotic', 'Necrotic']), df.transpose()[1:]],
        ignore_index=True,
        axis=1)
    mts = mts.append(dft, ignore_index=True)

alive.to_csv('ALIVE.csv', index=False, header=False)
apoptotic.to_csv('APOPTOTIC.csv', index=False, header=False)
necrotic.to_csv('NECROTIC.csv', index=False, header=False)
mts.to_csv('MTS.csv', index=False, header=False)
