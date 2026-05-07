
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

DATA_FOLDER="processed_output"
CLIENT_FILES=["client_1.csv","client_2.csv","client_3.csv","client_4.csv"]
OUTPUT_FOLDER="outputs"
os.makedirs(OUTPUT_FOLDER,exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/client_weights",exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/confusion_matrices",exist_ok=True)

LEARNING_RATE=0.01
LOCAL_EPOCHS=15
FEDERATED_ROUNDS=5
TEST_SIZE=0.2
RANDOM_STATE=42

def sigmoid(z):
    return 1/(1+np.exp(-z))

def init_weights(n):
    return np.zeros(n),0.0

def train_lr(X,y,w,b,lr,epochs):
    m=X.shape[0]
    # benign_count = np.sum(y == 0)
    # attack_count = np.sum(y == 1)

    # weight_benign = attack_count / benign_count
    # weight_attack = 1.0
    for _ in range(epochs):
        # sample_weights = np.where(
        #     y == 0,
        #     weight_benign,
        #     weight_attack
        # )
        pred=sigmoid(np.dot(X,w)+b)

        # errors = (pred - y) * sample_weights

        # dw = (1 / m) * np.dot(X.T, errors)
        # db = (1 / m) * np.sum(errors)
        dw=(1/m)*np.dot(X.T,(pred-y))
        db=(1/m)*np.sum(pred-y)
        w-=lr*dw
        b-=lr*db
    return w,b

def predict(X,w,b):
    return (sigmoid(np.dot(X,w)+b)>=0.5).astype(int)

def metrics(y,p):
    return {
        "accuracy":accuracy_score(y,p),
        "precision":precision_score(y,p,zero_division=0),
        "recall":recall_score(y,p,zero_division=0),
        "f1":f1_score(y,p,zero_division=0)
    }

def save_cm(y,p,title,path):
    cm=confusion_matrix(y,p)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm)
    fig,ax=plt.subplots(figsize=(5,5))
    disp.plot(ax=ax)
    plt.title(title)
    plt.savefig(path)
    plt.close()

def fedavg(local_weights,sizes):
    total=np.sum(sizes)
    avg_w=np.zeros_like(local_weights[0][0])
    avg_b=0.0
    for (w,b),s in zip(local_weights,sizes):
        avg_w+=(s/total)*w
        avg_b+=(s/total)*b
    return avg_w,avg_b

clients=[]

for file in CLIENT_FILES:
    df=pd.read_csv(f"{DATA_FOLDER}/{file}")
    df["Label"]=df["Label"].apply(lambda x:0 if x==0 else 1)

    X=df.drop(columns=["Label"]).values
    y=df["Label"].values

    Xtr,Xte,ytr,yte=train_test_split(
        X,y,test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    clients.append({
        "name":file.replace(".csv",""),
        "Xtr":Xtr,"Xte":Xte,
        "ytr":ytr,"yte":yte
    })

n_features=clients[0]["Xtr"].shape[1]
global_w,global_b=init_weights(n_features)

print("Starting Federated Training")

for r in range(FEDERATED_ROUNDS):
    local_weights=[]
    sizes=[]

    print(f"Round {r+1}")

    for client in clients:
        w=global_w.copy()
        b=global_b

        w,b=train_lr(
            client["Xtr"],
            client["ytr"],
            w,b,
            LEARNING_RATE,
            LOCAL_EPOCHS
        )

        local_weights.append((w,b))
        sizes.append(len(client["Xtr"]))

        np.save(
            f"{OUTPUT_FOLDER}/client_weights/{client['name']}_round_{r+1}.npy",
            w
        )

    global_w,global_b=fedavg(local_weights,sizes)

    np.save(
        f"{OUTPUT_FOLDER}/global_weights_round_{r+1}.npy",
        global_w
    )

all_true=[]
all_pred=[]

for client in clients:
    pred=predict(client["Xte"],global_w,global_b)

    m=metrics(client["yte"],pred)

    print(f"\n{client['name']}")
    print(m)

    all_true.extend(client["yte"])
    all_pred.extend(pred)

fed_metrics=metrics(np.array(all_true),np.array(all_pred))

with open(f"{OUTPUT_FOLDER}/federated_metrics.txt","w") as f:
    for k,v in fed_metrics.items():
        f.write(f"{k}: {v:.4f}\n")

save_cm(
    np.array(all_true),
    np.array(all_pred),
    "Federated Confusion Matrix",
    f"{OUTPUT_FOLDER}/confusion_matrices/federated_confusion_matrix.png"
)

print("\nStarting Centralized Baseline")

gdf=pd.read_csv(f"{DATA_FOLDER}/global_processed.csv")
gdf["Label"]=gdf["Label"].apply(lambda x:0 if x==0 else 1)

X=gdf.drop(columns=["Label"]).values
y=gdf["Label"].values

Xtr,Xte,ytr,yte=train_test_split(
    X,y,test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

#
# print("TRAIN DISTRIBUTION:")
# print(np.bincount(ytr))

# print("TEST DISTRIBUTION:")
# print(np.bincount(yte))
#

cw,cb=init_weights(Xtr.shape[1])

cw,cb=train_lr(
    Xtr,ytr,cw,cb,
    LEARNING_RATE,
    LOCAL_EPOCHS
)

cpred=predict(Xte,cw,cb)

#
# unique, counts = np.unique(cpred, return_counts=True)
# print(dict(zip(unique, counts)))
#

central_metrics=metrics(yte,cpred)

with open(f"{OUTPUT_FOLDER}/centralized_metrics.txt","w") as f:
    for k,v in central_metrics.items():
        f.write(f"{k}: {v:.4f}\n")

save_cm(
    yte,
    cpred,
    "Centralized Confusion Matrix",
    f"{OUTPUT_FOLDER}/confusion_matrices/centralized_confusion_matrix.png"
)

print("\nFederated Metrics")
print(fed_metrics)

print("\nCentralized Metrics")
print(central_metrics)

print("\nDone.")