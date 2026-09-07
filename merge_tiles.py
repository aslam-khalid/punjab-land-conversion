import rasterio
from rasterio.merge import merge
import glob

for year in ['2019', '2025']:
    tiles = sorted(glob.glob(f'data/raw/s2_faisalabad_{year}-*.tif'))
    print(f"Merging {len(tiles)} tiles for {year}: {tiles}")
    
    src_files = [rasterio.open(t) for t in tiles]
    mosaic, out_transform = merge(src_files)
    
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })
    
    out_path = f'data/raw/s2_faisalabad_{year}.tif'
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    for s in src_files:
        s.close()
    
    print(f"✅ Saved merged file: {out_path}")
