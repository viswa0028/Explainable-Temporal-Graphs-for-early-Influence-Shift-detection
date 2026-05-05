import pandas as pd

activity = pd.read_csv(
    "/Users/viswa/Desktop/PythonProject/Sna/data/higgs-activity_time.txt",
    sep=r"\s+",
    header=None,
    names=["user", "timestamp", "type"]
)

retweet = pd.read_csv(
    "/Users/viswa/Desktop/PythonProject/Sna/data/higgs-retweet_network.edgelist",
    sep=r"\s+",
    header=None,
    names=["source", "target"]
)

rt_activity = activity[activity["type"] == "RT"]

retweet_time = retweet.merge(
    rt_activity,
    left_on="source",
    right_on="user",
    how="inner"
)

retweet_time = retweet_time[["source", "target", "timestamp"]]
retweet_time.drop_duplicates(inplace=True)
print(retweet_time.head())
retweet_time["datetime"] = pd.to_datetime(
    retweet_time["timestamp"], unit="s"
)
print("hello")
retweet_time["date"] = retweet_time["datetime"].dt.date
print(retweet_time.head())

# print(retweet_time.head())
import networkx as nx

daily_graphs = {}

for day, group in retweet_time.groupby("date"):
    G = nx.DiGraph()
    for _, row in group.iterrows():
        G.add_edge(row["source"], row["target"])
    daily_graphs[day] = G

import pickle

with open("/Users/viswa/Desktop/PythonProject/Sna/data/daily_graphs.pkl", "wb") as f:
    pickle.dump(daily_graphs, f)

retweet_time.to_csv("/Users/viswa/Desktop/PythonProject/Sna/data/retweet_time.csv", index=False)