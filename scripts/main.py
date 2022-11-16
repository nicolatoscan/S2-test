# %%
from matplotlib import pyplot as plt
import numpy as np
import rasterio as rio
from rasterio import plot as rioplt
from skimage import exposure
import pandas as pd
from tqdm import tqdm
import random
import os
import json
import datetime
from sklearn.preprocessing import MultiLabelBinarizer

# %%
DATASET = '/home/toscan/dev/sat/dataset/BigEarthNet-v1.0'
files = os.listdir(DATASET)
bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
bandsNames = { 
    'B01': 'Coastal aerosol',
    'B02': 'Blue',
    'B03': 'Green',
    'B04': 'Red',
    'B05': 'Vegetation red edge 1',
    'B06': 'Vegetation red edge 2',
    'B07': 'Vegetation red edge 3',
    'B08': 'NIR',
    'B8A': 'Vegetation Red Edge',
    'B09': 'Water vapor',
    'B10': 'Cirrus',
    'B11': 'SWIR 1',
    'B12': 'SWIR 2'
}
name = "S2A_MSIL2A_20171104T095201_5_43"

# %%
def getFileInfo():
    filesInfo = []
    for name in tqdm(files):
        with open(f"{DATASET}/{name}/{name}_labels_metadata.json") as f:
            d = json.load(f)
            [ SentinelId, MSIL2A, filedate, h, v ] = name.split('_')

            filesInfo.append([
                name,
                SentinelId,
                MSIL2A,
                datetime.datetime.fromisoformat(d['acquisition_date']),
                int(h),
                int(v),
                pd.Series(d['labels']),
                int(d['coordinates']['ulx']),
                int(d['coordinates']['uly']),
                int(d['coordinates']['lrx']),
                int(d['coordinates']['lry']),
            ])
    return filesInfo
dff = pd.DataFrame(getFileInfo(), columns=['name', 'SentinelId', 'MSIL2A', 'date', 'h', 'v', 'labels', 'ulx', 'uly', 'lrx', 'lry'])
mlb = MultiLabelBinarizer()
onehot = dff.join(pd.DataFrame(mlb.fit_transform(dff.pop('labels')),
                          columns=mlb.classes_,
                          index=dff.index))
# dff.to_csv('files.csv', index=False)

# %%
def sameCoor(ulx, uly):
    return dff[ (dff['ulx'] == ulx) & (dff['uly'] == uly) ]['name'].to_list()

def getRgbMatrix(name):
    rgb = np.dstack([src.read(1) for src in [ rio.open(f"{DATASET}/{name}/{name}_{b}.tif") for b in [ 'B04', 'B03', 'B02' ] ] ])
    return exposure.rescale_intensity(rgb, in_range=tuple(np.percentile(rgb, (2,98)))) / 100000

def plotRGB(name):
    plt.title('RGB')
    plt.axis('off')
    plt.imshow(getRgbMatrix(name))
    plt.show()
# plotRGB("S2A_MSIL2A_20170613T101031_37_57")

def plotRgbList(names):
    fig, axs = plt.subplots(len(names), figsize=(5, 5 * len(names)))
    for i, name in enumerate(names):
        axs[i].set_title(name.split('_')[2].split('T')[0])
        axs[i].axis('off')
        axs[i].imshow(getRgbMatrix(name))

def plotBands(name):
    fig, axs = plt.subplots(3, 4, figsize=(15, 15))
    for i, b in enumerate(bands):
        rioplt.show(rio.open(f"{DATASET}/{name}/{name}_{b}.tif"), ax=axs[i//4, i%4], cmap='gray')
        axs[i//4, i%4].set_title(bandsNames[b])
        axs[i//4, i%4].axis('off')
    plt.show()
# plotBands("S2A_MSIL2A_20170613T101031_37_57")

# %%
n = 10000
choosenFiles = random.sample(files, n)

# %% fill dataframe with stats
lines = []
for name in tqdm(choosenFiles):
    for i, b in enumerate(bands):
        data = rio.open(f"{DATASET}/{name}/{name}_{b}.tif").read(1)
        min = data.min()
        max = data.max()
        lines.append([
            b,
            data.mean(),
            data.std(),
            min,
            max,
            max - min,
            np.percentile(data, 2),
            np.percentile(data, 98)
        ])
df = pd.DataFrame(lines, columns=['band', 'mean', 'std', 'min', 'max', 'range', 'p2', 'p98'])
# %% show df
# pd.options.display.float_format = "{:,.2f}".format
statsDf = df.groupby('band').agg({
    'mean': ['mean', 'std'],
    'std': ['mean', 'std'],
    'min': ['min'],
    'max': ['max'],
    'range': ['mean', 'std', lambda x: np.percentile(x, q = 2) , lambda x: np.percentile(x, q = 95) ],
    'p2': ['mean', 'std'],
    'p98': ['mean', 'std']
})
display(statsDf)


# %% hist
p2 = df.groupby('band').agg(np.mean)['p2']
p98 = df.groupby('band').agg(np.mean)['p98']
min = df.groupby('band').agg(np.min)['min']
max = df.groupby('band').agg(np.max)['max']

histBins = 200
hists = np.zeros((len(bands), histBins), dtype=int)
lines = []
limits = { b: (min[b], max[b]) for b in bands }
for name in tqdm(choosenFiles):
    for i, b in enumerate(bands):
        data = rio.open(f"{DATASET}/{name}/{name}_{b}.tif").read(1)
        hists[i] += np.histogram(data, bins=histBins, range=limits[b])[0]


# %% plot hists bar chart
fig, axs = plt.subplots(6, 2, figsize=(10, 20))
fig.suptitle(f'Band histograms - Bins = {histBins}')
for i, b in enumerate(bands):
    axs[i//2, i%2].bar(range(histBins), hists[i])
    axs[i//2, i%2].set_title(f"{bandsNames[b]} | {min[b]}-{max[b]}")
plt.show()

# %% band ranges
rangeBins = 100
fig, axs = plt.subplots(6, 2, figsize=(10, 20))
fig.suptitle(f'Band ranges - Bins = {rangeBins}')
for i, b in enumerate(bands):
    axs[i//2, i%2].hist(df[df['band'] == b]['range'], bins=rangeBins)
    axs[i//2, i%2].set_title(f"{bandsNames[b]}")
plt.show()
# %%
