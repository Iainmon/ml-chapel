import os
import numpy as np
import cv2
import struct


translate = {'cane': 'dog', 'cavallo': 'horse', 'elefante': 'elephant', 'farfalla': 'butterfly', 'gallina': 'chicken', 'gatto': 'cat', 'mucca': 'cow', 'pecora': 'sheep', 'scoiattolo': 'squirrel', 'dog': 'cane', 'cavallo': 'horse', 'elephant' : 'elefante', 'butterfly': 'farfalla', 'chicken': 'gallina', 'cat': 'gatto', 'cow': 'mucca', 'spider': 'ragno', 'squirrel': 'scoiattolo'}
translate_inv = {v: k for k, v in translate.items()}

folers = os.listdir('raw-img')
categories = {}
for folder in folers:
    if folder == '.DS_Store': continue
    key = (dict(translate_inv,**translate))[folder]
    categories[key] = ['raw-img/' + folder + '/' + p for p in os.listdir('raw-img/' + folder)]

def centering_image(img):
    size = [256,256]

    img_size = img.shape[:2]

    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized

total_images = sum(map(len,categories.values()))


images = []
image_categories_idx_map = []
for i, (category,file_path) in enumerate([(k,v) for k,vs in categories.items() for v in vs]):
    #read image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_size = 128
    #resize
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*img_size/img.shape[0]),img_size)
    else:
        tile_size = (img_size, int(img.shape[0]*img_size/img.shape[1]))

    #centering
    img = centering_image(cv2.resize(img, dsize=tile_size))

    #out put 224*224px 
    img = img[16:144, 16:144]
    images.append(img)
    image_categories_idx_map.append(category)

images = np.array(images)

def write_chpl_tensor(a,file):
    shape = a.shape
    shape_size = len(shape)
    file.write(struct.pack('<q', shape_size))
    for s in shape:
        file.write(struct.pack('<q', s))
    # file.write(a.tobytes())
    for x in np.nditer(a):
        file.write(struct.pack('<d', x))

consolidated = {}
for i in range(total_images):
    category = image_categories_idx_map[i]
    img = images[i,:,:,:]
    if category not in consolidated.keys():
        consolidated[category] = []
    if len(consolidated[category]) > 500:
        continue
    consolidated[category].append(img)

consolidated = {k: np.array(v) for k,v in consolidated.items()}

for category, imgs in consolidated.items():
    with open(f'export/{category}.bin', 'wb') as f:
        write_chpl_tensor(imgs,f)
        print(f'export/{category}.bin')
