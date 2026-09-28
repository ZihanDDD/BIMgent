import json


def load_json(file_path):
    with open(file_path, mode='r', encoding='utf8') as fp:
        return json.load(fp)


def save_json(file_path, json_dict, indent=4):
    with open(file_path, mode='w', encoding='utf8') as fp:
        json.dump(json_dict, fp, ensure_ascii=False, indent=indent, default=str)
