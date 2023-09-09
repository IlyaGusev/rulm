import sys
import json
import random
from collections import Counter

from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

input_path = sys.argv[1]
y_pred = []
y_true = []
records = []
with open(input_path) as r:
    for line in r:
        r = json.loads(line)
        if not r["label"]:
            continue
        records.append(r)
        prediction = r["prediction"]
        if "1" in prediction:
            y_pred.append(-1)
        elif "2" in prediction:
            y_pred.append(1)
        else:
            y_pred.append(0)
        y_true.append(max(min(int(r["label"]), 1), -1))

cnt = 0
for i1, i2, r in zip(y_pred, y_true, records):
    if i1 + i2 == 0 and i1 != 0:
        print(r)
        cnt += 1
print(cnt)
print(Counter(y_pred))
print(Counter(y_true))

print(Counter(y_pred)[1] / (Counter(y_pred)[-1] + Counter(y_pred)[1]))
print(Counter(y_true)[1] / (Counter(y_true)[-1] + Counter(y_pred)[1]))

print("File")
print(accuracy_score(y_pred, y_true))
print(pearsonr(y_pred, y_true).statistic)

y_pred = [random.randint(0, 2) - 1 for i in range(len(y_true))]
print("Random")
print(accuracy_score(y_pred, y_true))
print(pearsonr(y_pred, y_true).statistic)
