import os
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scikit_posthocs as sp
import scipy.stats as stats

VERSION = "v1"
WANDB_USER = "erhaemael-politeknik-negeri-bandung"
DATASETS = ["wesad"]  
METRICS = {"units": ["f1_score_ar3"]}
AUCs = {"units": ["AUC_ROC", "AUC_PR"]}
RUNS = {"units": 25}

api = wandb.Api()

def process_project(project_name):
    if "wesad" not in project_name:
        return
    if VERSION not in project_name:
        return
    
    runs = api.runs(f"{WANDB_USER}/{project_name}")
    model = "units"
    metrics = METRICS[model]
    aucs = AUCs[model]
    
    if len(runs) != RUNS[model] * 2:
        print(f"Skipping {project_name} due to missing runs ({len(runs)}/{RUNS[model] * 2})")
        return
    
    best_f1s, best_aucrocs, best_aucprs = [], [], []
    
    for run in tqdm(runs, desc=f"Processing {project_name}"):
        data = pd.concat([pd.DataFrame(run.scan_history(keys=[metric])) for metric in metrics + aucs], axis=1)
        
        best_idx = data[metrics[0]].idxmax()
        best_f1s.append(float(data.loc[best_idx, metrics[0]]))
        best_aucrocs.append(float(data.loc[best_idx, aucs[0]]))
        best_aucprs.append(float(data.loc[best_idx, aucs[1]]))
    
    avg_f1 = np.mean(best_f1s)
    std_f1 = np.std(best_f1s)
    avg_aucroc = np.mean(best_aucrocs)
    avg_aucpr = np.mean(best_aucprs)
    
    print(f"Project: {project_name} - F1: {avg_f1}, AUC_ROC: {avg_aucroc}, AUC_PR: {avg_aucpr}")
    
    with open("results_wesad.csv", "a") as csv:
        csv.write(f"{project_name},{avg_f1},{avg_aucroc},{avg_aucpr},{len(runs)}\n")
    
    return avg_f1, avg_aucroc, avg_aucpr

def plot_step_projects():
    if "results_wesad.csv" not in os.listdir():
        print("No results file found for plotting.")
        return
    
    df = pd.read_csv("results_wesad.csv", header=None, names=["Project", "F1", "AUC_ROC", "AUC_PR", "Runs"])
    df["Step"] = df["Project"].str.extract(r'step(\d+)').astype(float)
    df = df.dropna().sort_values("Step")
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["Step"], df["F1"], marker='o', label="F1 Score")
    plt.fill_between(df["Step"], df["F1"] - df["F1"].std(), df["F1"] + df["F1"].std(), alpha=0.2)
    plt.xlabel("Step")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig("step_projects_wesad.svg")
    plt.show()

def main():
    if "results_wesad.csv" in os.listdir():
        os.remove("results_wesad.csv")
    
    projects = api.projects(f"{WANDB_USER}")
    for project in tqdm(projects, desc="Processing projects"):
        process_project(project.name.lower())
    
    plot_step_projects()

if __name__ == "__main__":
    main()
