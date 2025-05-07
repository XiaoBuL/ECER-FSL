import numpy as np
import base64
import os
import io
import json
import time
import pandas as pd

import os
from model.clip import clip
import torch
import pandas as pd
import torch.nn.functional as F
model, preprocess = clip.load('ViT-B/32', 'cuda')
device = "cuda" if torch.cuda.is_available() else "cpu"





def get_selected_entity(class_name,entity_lists,K):
    with torch.no_grad():
        selected_entity = {}
        for key in class_name:
            clip_name = class_name[key]
            entity_list = entity_lists[key]
            sim_score = []
            text_inputs_entity = torch.cat([clip.tokenize(f"{c}") for c in entity_list]).to(device)
            text_features_entity = model.encode_text(text_inputs_entity)#N*512
            
            text_inputs_clipname = torch.cat([clip.tokenize(f"{c}") for c in clip_name]).to(device)
            text_features_clipname = model.encode_text(text_inputs_clipname) # 1*512
            
            sim = F.cosine_similarity(text_features_clipname, text_features_entity, dim=1)
            sorted_indices = torch.argsort(sim, descending=True)[:K]
            entitys = []
            for i in sorted_indices:
                entitys.append(entity_list[i])
            selected_entity[key] = entitys
        return  selected_entity



if __name__ == '__main__':  
    filename = 'path/to/yourcode/PVSA/sem_json/miniImageNet_entities.json'
    with open(filename, 'r') as f:
        mini_entity = json.load(f)
    class_name = {}
    entity_list = {}
    for key in mini_entity:
        class_name[key] = ['a photo of a(n) {}'.format(str(key))]
        entity_list[key] = mini_entity[key]['relevant_entities']

    K=3
    selected_entity = get_selected_entity(class_name,entity_list,K)
    print(selected_entity)