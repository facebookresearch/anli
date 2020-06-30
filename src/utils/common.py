# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

import json
from json import JSONEncoder
from tqdm import tqdm
import config


registered_jsonabl_classes = {}

# Some Jsonable classes, for easy json serialization.


def register_class(cls):
    global registered_jsonabl_classes
    if cls not in registered_jsonabl_classes:
        registered_jsonabl_classes.update({cls.__name__: cls})


class JsonableObj(object):
    pass


class JsonableObjectEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, JsonableObj):
            d = {'_jcls_': type(o).__name__}
            d.update(vars(o))
            return d
        else:
            return super().default(o)


def unserialize_JsonableObject(d):
    global registered_jsonabl_classes
    classname = d.pop('_jcls_', None)
    if classname:
        cls = registered_jsonabl_classes[classname]
        obj = cls.__new__(cls)              # Make instance without calling __init__
        for key, value in d.items():
            setattr(obj, key, value)
        return obj
    else:
        return d


def json_dumps(item):
    return json.dumps(item, cls=JsonableObjectEncoder)


def json_loads(item_str):
    return json.loads(item_str, object_hook=unserialize_JsonableObject)

# Json Serializable object finished.


def save_jsonl(d_list, filename):
    print("Save to Jsonl:", filename)
    with open(filename, encoding='utf-8', mode='w') as out_f:
        for item in d_list:
            out_f.write(json.dumps(item, cls=JsonableObjectEncoder) + '\n')


def load_jsonl(filename, debug_num=None):
    d_list = []
    with open(filename, encoding='utf-8', mode='r') as in_f:
        print("Load Jsonl:", filename)
        for line in tqdm(in_f):
            item = json.loads(line.strip(), object_hook=unserialize_JsonableObject)
            d_list.append(item)
            if debug_num is not None and 0 < debug_num == len(d_list):
                break

    return d_list


def load_json(filename, **kwargs):
    with open(filename, encoding='utf-8', mode='r') as in_f:
        return json.load(in_f, object_hook=unserialize_JsonableObject, **kwargs)


def save_json(obj, filename, **kwargs):
    with open(filename, encoding='utf-8', mode='w') as out_f:
        json.dump(obj, out_f, cls=JsonableObjectEncoder, **kwargs)
        out_f.close()