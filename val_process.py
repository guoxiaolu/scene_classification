import os
import shutil
from tqdm import tqdm

categories = '/media/wac/backup/places365_Standard/filelist_places365-standard/categories_places365_pure.txt'
val_file = '/media/wac/backup/places365_Standard/filelist_places365-standard/places365_val.txt'
val_img_path = '/media/wac/backup/places365_Standard/val_256'
dst_path = '/media/wac/backup/places365_Standard/val_data'

f = open(categories, mode='r')
classes = [line.strip().split(' ')[0] for line in f.readlines()]
f.close()

f = open(val_file, mode='r')
for line in tqdm(f.readlines()):
    content = line.strip().split(' ')
    src_name = os.path.join(val_img_path, content[0])
    dst_category_path = os.path.join(dst_path, classes[int(content[1])])
    if not os.path.exists(dst_category_path):
        os.mkdir(dst_category_path)
    dst_name = os.path.join(dst_category_path, content[0])
    shutil.copy(src_name, dst_name)
f.close()