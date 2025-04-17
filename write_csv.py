import csv
import torch
import os

root = "results/"
envs = os.listdir(root)
print(envs)

data = {}
for env in envs:
    base_path = f"{root}{env}"
    files = os.listdir(base_path)

    for file in files:
        path = f"{base_path}/{file}"
        if os.path.isdir(path):
            temp = os.listdir(path)
            files.extend( [f"{file}/{file2}" for file2 in temp])
        else:
            run_name = path.split("/")[-1]
            data[f"{env}+{run_name}"] = torch.load(path)
            data[f"{env}+{run_name}"]["run_name"] = f"{env}_{run_name}"

data2 = []
for row in data:
    data2.append(data[row])
    print(data[row])

with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['asr', 'asr_std', 'return', 'return_std', "run_name"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data2)

