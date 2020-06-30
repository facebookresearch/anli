# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

import uuid


def list_to_dict(d_list, key_fields):       #   '_id' or 'pid'
    d_dict = dict()
    for item in d_list:
        assert key_fields in item
        d_dict[item[key_fields]] = item
    return d_dict


def dict_to_list(d_dict):
    d_list = []
    for key, value in d_dict.items():
        d_list.append(value)
    return d_list


def append_item_from_dict_to_list(d_list, d_dict, key_fieldname, append_fieldnames):
    if not isinstance(append_fieldnames, list):
        append_fieldnames = [append_fieldnames]
    for item in d_list:
        key = item[key_fieldname]
        if key in d_dict:
            for append_fieldname in append_fieldnames:
                item[append_fieldname] = d_dict[key][append_fieldname]
        else:
            print(f"Potential Error: {key} not in scored_dict. Maybe bc all forward items are empty.")
            for append_fieldname in append_fieldnames:
                item[append_fieldname] = []
    return d_list


def append_item_from_dict_to_list_hotpot_style(d_list, d_dict, key_fieldname, append_fieldnames):
    if not isinstance(append_fieldnames, list):
        append_fieldnames = [append_fieldnames]
    for item in d_list:
        key = item[key_fieldname]
        for append_fieldname in append_fieldnames:
            if key in d_dict[append_fieldname]:
                item[append_fieldname] = d_dict[append_fieldname][key]
            else:
                print(f"Potential Error: {key} not in scored_dict. Maybe bc all forward items are empty.")
                # for append_fieldname in append_fieldnames:
                item[append_fieldname] = []
    return d_list


def append_subfield_from_list_to_dict(subf_list, d_dict, o_key_field_name, subfield_key_name,
                                      subfield_name='merged_field', check=False):
    # Often times, we will need to split the one data point to multiple items to be feeded into neural networks
    # and after we obtain the results we will need to map the results back to original data point with some keys.

    # This method is used for this purpose.
    # The method can be invoke multiple times, (in practice usually one batch per time.)
    """
    :param subf_list:               The forward list.
    :param d_dict:                  The dict that contain keys mapping to original data point.
    :param o_key_field_name:        The fieldname of original data point key. 'pid'
    :param subfield_key_name:       The fieldname of the sub item. 'fid'
    :param subfield_name:           The merge field name.       'merged_field'
    :param check:
    :return:
    """
    for key in d_dict.keys():
        d_dict[key][subfield_name] = dict()

    for item in subf_list:
        assert o_key_field_name in item
        assert subfield_key_name in item
        map_id = item[o_key_field_name]
        sub_filed_id = item[subfield_key_name]
        assert map_id in d_dict

        # if subfield_name not in d_dict[map_id]:
        #     d_dict[map_id][subfield_name] = dict()

        if sub_filed_id not in d_dict[map_id][subfield_name]:
            if check:
                assert item[o_key_field_name] == map_id
            d_dict[map_id][subfield_name][sub_filed_id] = item
        else:
            print("Duplicate forward item with key:", sub_filed_id)

    return d_dict


if __name__ == '__main__':
    oitems = []
    for i in range(3):
        oitems.append({'_id': i})

    fitems = []
    for item in oitems:
        oid = item['_id']
        for i in range(int(oid) + 1):
            fid = str(uuid.uuid4())
            fitems.append({
                'oid': oid,
                'fid': fid,
            })

    o_dict = list_to_dict(oitems, '_id')
    append_subfield_from_list_to_dict(fitems, o_dict, 'oid', 'fid', check=True)

    print(fitems)
    print(o_dict)
