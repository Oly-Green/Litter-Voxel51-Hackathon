THREAT_LEVELS = {4: ["styrofoam"], 3: ["plastic"], 2: ["glass"], 1:["paper"]}
# threat_levels = {4: ["styrofoam"]}
# super_cat_lists_consolidated = {"styrofoam": [57]}

import json
import os
from tqdm import tqdm

def parse_supercategory(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            annotations_dict = json.load(f)

    super_cat_lists = {}

    for cat in annotations_dict["categories"]:
        if not cat["supercategory"] in super_cat_lists:
            super_cat_lists[cat["supercategory"].lower()] = [cat["id"]]
        else:
            super_cat_lists[cat["supercategory"]].append(cat["id"])

    keywords = ["styrofoam", "plastic", "glass", "paper"]
    super_cat_lists_consolidated = {}

    for cat_name in super_cat_lists.keys():
        for word in keywords:
            if word in cat_name:
                if word not in super_cat_lists_consolidated.keys():
                    super_cat_lists_consolidated[word] = super_cat_lists[cat_name]
                else:
                    super_cat_lists_consolidated[word].extend(super_cat_lists[cat_name])

    return super_cat_lists_consolidated


file_name = "annotations_unofficial.json"
path = os.path.join("C:/Users/olive/Documents/NortheasternDocuments/Voxel51-Hackathon/TACO/data", file_name)
super_cats = parse_supercategory(path)
print(super_cats)
