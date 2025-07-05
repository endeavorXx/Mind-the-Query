import os, json
from typing import List, Dict, Any
import re 
import random
import numpy as np

class DataManager:
    def __init__(self, main_folder_name = None, data_folder_name = None, prompt_version = None, database = None, temperature = 0.0):
       if main_folder_name is not None:
        os.makedirs(f"{main_folder_name}", exist_ok=True)
        os.makedirs(f"{main_folder_name}/{database}/t_{temperature}/{data_folder_name}_{prompt_version}/cypher_pairs_with_reason", exist_ok=True)
        os.makedirs(f"{main_folder_name}/{database}/t_{temperature}/{data_folder_name}_{prompt_version}/cypher_pairs_only", exist_ok=True)
        self.save_main_path = f"{main_folder_name}/{database}/t_{temperature}/{data_folder_name}_{prompt_version}/cypher_pairs_with_reason"
        self.suppl_path = f"{main_folder_name}/{database}/t_{temperature}/{data_folder_name}_{prompt_version}/cypher_pairs_only"
    

    def get_suppl_data(self, data):
        suppl_data = []
        for i in data:
            tmp_dict = {}
            if (type(i) == dict):
                tmp_dict["NL Question"] = i["NL Question"]
                tmp_dict["Cypher"] = i["Cypher"]
            if (type(i) == list):
                suppl_data.append(i)
            suppl_data.append(tmp_dict)

        return suppl_data
    
    def save_data(self, data, file_name):
        with open(f"{self.save_main_path}/{file_name}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        print(f"Data has been successfully saved to {self.save_main_path}/{file_name}")

        suppl_data = self.get_suppl_data(data)
        with open(f"{self.suppl_path}/{file_name}.json", 'w', encoding='utf-8') as f:
            json.dump(suppl_data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
    
    def get_sampled_data(self, node_data, rels_data):
        """
        Returns a random sample of data entries.
        """
        data = {
            "nodes": node_data,
            "relationships": rels_data
        }
        return data