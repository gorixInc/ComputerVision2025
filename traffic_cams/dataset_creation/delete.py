# %%
from glob import glob
import os
# %%
for i in range(47, 59):
    imgs = glob(f'downloaded_images/2025-03-23/*16{i}??.jpg')
    for img in imgs:
        print(img)
        os.remove(img)
# %%
