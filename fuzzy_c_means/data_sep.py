import pandas as pd

dataset = pd.read_csv("dataset.csv", header=None)

index = 1
mining_data = []
classification_data = []
for row in dataset.values:
    if index%5 == 0:
        classification_data.append(row)
    else:
        mining_data.append(row)
    index += 1

mining_data = pd.DataFrame(mining_data)
classification_data = pd.DataFrame(classification_data)

mining_data.to_csv("mining_data.csv", header=False, index=False)
classification_data.to_csv("classification_data.csv", header=False, index=False)

