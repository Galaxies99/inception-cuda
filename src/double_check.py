import json
import numpy as np

with open('../data/inceptionV3.json', 'r') as fp:
    std = json.load(fp)

print("std read finish")

with open('../data/inceptionV3-check.json', 'r') as fp:
    out = json.load(fp)

print("out read finish")

for key in std.keys():
    print("checking", key, "...")
    if key not in out.keys():
        print('[Error] Key', key, 'not in standard keys.')
    for key2 in std[key].keys():
        if key2 not in out[key].keys():
            print('[Error] Secondary key', key2, 'not in standard keys.')
        std_val = std[key][key2]
        out_val = out[key][key2]
        if type(std_val) != list:
            std_val = [std_val]
        if type(out_val) != list:
            out_val = [out_val]
        std_np = np.array(std_val).flatten()
        out_np = np.array(out_val).flatten()
        if std_np.shape[0] != out_np.shape[0]:
            print('[Error] Shape mismatch.');
        len = std_np.shape[0]
        for k in range(len):
            if(abs(std_np[k] - out_np[k]) > 1e-5):
                print('[Error] value differs!')

        