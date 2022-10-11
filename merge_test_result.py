import os
import re
import numpy as np
import json
import argparse


# parse required arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path", nargs="+", type=str, help="path of prediction files")
parser.add_argument("--num_crop", default=3, type=int, help="number of spatial crops for each test video")
parser.add_argument("--annotation_file", type=str,
                    default="/data/shared/ego4d/v1/annotations/fho_oscc-pnr_test_unannotated.json" ,
                    help="path of test annotation file"
                    )

args = parser.parse_args()


path = args.path

output_path = "/".join(args.path[0].split("/")[:-1])
num_crop = args.num_crop
annotation_file = args.annotation_file

pattern_twohead = "(.*?) \[(.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?)\] \[(.*?), (.*?)\] (\d) \[(.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?)\]"
pred_dict = {}

for p in path:
    # find all results
    raw = open(p, "r").read()

    results = re.findall(pattern_twohead, raw)

    # wash results
    for result in results:
        id = result[0]

        loc_preds = result[1:18]
        cls_pred = result[18:20]
        crop_num = result[20]
        frame_index = result[21:]

        if id not in pred_dict.keys():
            pred_dict[id] = {
                0:{},
                1:{},
                2:{},
            }
        pred_dict[id][int(crop_num)] = {
            "loc": [float(pred) for pred in loc_preds],
            "cls": [float(pred) for pred in cls_pred],
            "idx": [int(idx) for idx in frame_index]
        }

# combine results
final_preds = {}
for k,v in pred_dict.items():
    loc = []
    idx = []
    cls = 0
    for i in range(num_crop):
        try:
            pos = np.argmax(v[i]["loc"])
            loc.append(pos)
            idx.append(v[i]["idx"][pos] if pos != 16 else -1)
            cls += np.argmax(v[i]["cls"])
        except:
            # in case some predictions are missing
            print(f"{i}th crop of {k} do not exist ")
            continue

    final_preds[k] = [loc, idx, cls/num_crop]


# save results to json file

cls_final = []
for k, v in final_preds.items():
    cls_final.append({
        "unique_id": k,
        "state_change": True if v[2] > 0.5 else False,
    })

clip_rawinfo = json.load(open(annotation_file))["clips"]
clip_dict = {}

for clip in clip_rawinfo:
    id = clip["unique_id"]
    clip_dict[id] = {
        "sf": int(clip["parent_start_frame"]),
        "ef": int(clip["parent_end_frame"])
    }

loc_final = []
for k, v in final_preds.items():
    loc_np = np.array(v[0])
    idx = np.array(v[1])

    pnr_idx = []
    for i in range(num_crop):
        if idx[i] == -1:
            continue
        pnr_idx.append(idx[i])

    # print(pnr_idx)
    if len(pnr_idx) == 0:
        pnr = 0.4 * (clip_dict[k]["ef"] - clip_dict[k]["sf"])
    else:
        pnr = sum(pnr_idx)/len(pnr_idx) - clip_dict[k]["sf"]

    loc_final.append({
        "unique_id": k,
        "pnr_frame":  pnr
    })


cls_bar = open(os.path.join(output_path, "cls_final.json"), "w")
cls_bar.write(json.dumps(cls_final))

loc_bar = open(os.path.join(output_path, "loc_final.json"), "w")
loc_bar.write(json.dumps(loc_final))
