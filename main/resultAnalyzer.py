import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from util import readFromCsv, evaluateAdaptations


font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)


def personalizedBoxPlot(data, name, columnNames=None, percentage=False, path=None, show=False,
                        seconds=False, legendInside=False, logscale=False,
                        algorithms=("NSGA-III", "XDA", "Anchors", "WIP")):

    columns = data.columns
    nColumns = len(columns)
    nAlgorithms = len(algorithms)

    print("Columns:", columns)
    print("Data shape:", data.shape)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # create boxplot
    bp = ax1.boxplot(
        [data[col].dropna().values for col in columns],
        patch_artist=True,
        notch=True,
        vert=True
    )

    # generate colors (safe version)
    base_colors = plt.cm.Spectral(np.linspace(.1, .9, nAlgorithms))

    colors = []
    for i in range(nColumns):
        colors.append(base_colors[i % nAlgorithms])

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B', linewidth=1.5, linestyle=":")

    # caps
    for cap in bp['caps']:
        cap.set(color='#8B008B', linewidth=2)

    # medians
    for median in bp['medians']:
        median.set(color='red', linewidth=3)

    # fliers
    for flier in bp['fliers']:
        flier.set(marker='D', color='#e7298a', alpha=0.5)

    if logscale:
        ax1.set_yscale('log')

    # ----- X axis (grouped by requirements) -----
    if columnNames is not None and len(columnNames) > 0:

        groupSize = nAlgorithms
        nGroups = len(columnNames)

        positions = np.arange(1, nGroups * groupSize + 1)

        centers = [
            np.mean(positions[i * groupSize:(i + 1) * groupSize])
            for i in range(nGroups)
        ]

        ax1.set_xticks(centers)
        ax1.set_xticklabels(columnNames)

    else:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # ----- Y axis -----
    if percentage:
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

        if (data.max().max() - data.min().min()) / 8 < 0.01:
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

    if seconds:
        def y_fmt(x, y):
            return str(int(x)) + ' s' if x >= 1 else str(x) + ' s'
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

    # ----- legend -----
    box = ax1.get_position()
    ax1.set_position([
        box.x0,
        box.y0 + box.height * 0.1,
        box.width,
        box.height * 0.9
    ])

    legend_handles = [bp["boxes"][i] for i in range(nAlgorithms)]

    if legendInside:
        ax1.legend(legend_handles, algorithms)
    else:
        ax1.legend(
            legend_handles,
            algorithms,
            ncol=nAlgorithms,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.1)
        )

    # title
    plt.title(name)

    # remove top/right ticks
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    plt.tight_layout()

    if path is not None:
        plt.savefig(path + name)

    if show:
        fig.show()
    else:
        plt.clf()

def personalizedBarChart(data, name, path=None, show=False, percentage=False):

    # convert Series to DataFrame (needed for iteration plots)
    if isinstance(data, pd.Series):
        data = data.to_frame()

    nAlgorithms = len(data.columns)

    colors = plt.cm.Spectral(np.linspace(.1, .9, nAlgorithms))

    fig, ax = plt.subplots()

    data.plot.bar(ax=ax, title=name, color=colors)

    # x-axis
    if len(data.index) > 1:
        plt.xticks(rotation=0)
    else:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # y-axis
    ax.set_ylim(0, 1)

    if percentage:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    # bar labels
    for container in ax.containers:

        if percentage:
            values = ['{:.1%}'.format(v) for v in container.datavalues]
        else:
            values = ['{:.2f}'.format(v) for v in container.datavalues]

        ax.bar_label(container, labels=values, fontsize=10)

    plt.tight_layout()

    if path is not None:
        plt.savefig(path + name)

    if show:
        plt.show()
    else:
        plt.close()

    # number of algorithms (columns)
    nAlgorithms = len(data.columns)

    # dynamic colors
    colors = plt.cm.Spectral(np.linspace(.1, .9, nAlgorithms))

    fig, ax = plt.subplots()

    data.plot.bar(ax=ax, title=name, color=colors)

    # x-axis labels
    if len(data.index) > 1:
        plt.xticks(rotation=0)
    else:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # y-axis
    ax.set_ylim(0, 1)

    if percentage:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    # bar labels
    for container in ax.containers:

        if percentage:
            values = ['{:.1%}'.format(v) for v in container.datavalues]
        else:
            values = ['{:.2f}'.format(v) for v in container.datavalues]

        ax.bar_label(container, labels=values, fontsize=10)

    plt.tight_layout()

    # save figure
    if path is not None:
        plt.savefig(path + name)

    if show:
        plt.show()
    else:
        plt.close()

os.chdir(sys.path[0])
evaluate = False

pathToResults = "../results/" #sys.argv[1]

featureNames = ["cruise speed",
                "image resolution",
                "illuminance",
                "controls responsiveness",
                "power",
                "smoke intensity",
                "obstacle size",
                "obstacle distance",
                "firm obstacle"]

reqs = ["req_0", "req_1", "req_2", "req_3"]
reqsNamesInGraphs = ["R1", "R2", "R3", "R4"]

# read dataframe from csv
results = readFromCsv(pathToResults + 'results_new.csv')
nReqs = len(results["nsga3_confidence"][0])
reqs = reqs[:nReqs]
reqsNamesInGraphs = reqsNamesInGraphs[:nReqs]
targetConfidence = np.full((1, nReqs), 0.8)[0]

if evaluate:
    evaluateAdaptations(results, featureNames)

# read outcomes from csv
customOutcomes = pd.read_csv(pathToResults + 'customDataset.csv')
nsga3Outcomes = pd.read_csv(pathToResults + 'nsga3Dataset.csv')
anchorsOutcomes = pd.read_csv(pathToResults + 'anchorsDataset.csv')
wipOutcomes = pd.read_csv(pathToResults + 'wipDataset.csv')

# build indices arrays
nsga3ConfidenceNames = ['nsga3_confidence_' + req for req in reqs]
nsga3OutcomeNames = ['nsga3_outcome_' + req for req in reqs]
customConfidenceNames = ['custom_confidence_' + req for req in reqs]
customOutcomeNames = ['custom_outcome_' + req for req in reqs]
anchorsConfidenceNames = ['anchors_confidence_' + req for req in reqs]
anchorsOutcomeNames = ['anchors_outcome_' + req for req in reqs]
wipConfidenceNames = ['wip_confidence_' + req for req in reqs]
wipOutcomeNames = ['wip_outcome_' + req for req in reqs]

#outcomes dataframe
outcomes = pd.concat([nsga3Outcomes[reqs], customOutcomes[reqs], anchorsOutcomes[reqs], wipOutcomes[reqs]], axis=1)
#outcomes.columns = np.append(nsga3OutcomeNames, customOutcomeNames, anchorsOutcomeNames)
outcomes.columns = np.array(nsga3OutcomeNames + customOutcomeNames + anchorsOutcomeNames + wipOutcomeNames)

outcomes = outcomes[list(sum(zip(nsga3OutcomeNames, customOutcomeNames, anchorsOutcomeNames, wipOutcomeNames), ()))]

# decompose arrays columns into single values columns
nsga3Confidences = pd.DataFrame(results['nsga3_confidence'].to_list(),
                                columns=nsga3ConfidenceNames)
customConfidences = pd.DataFrame(results['custom_confidence'].to_list(),
                                 columns=customConfidenceNames)
anchorsConfidences = pd.DataFrame(results['anchors_confidence'].to_list(),
                                  columns=anchorsConfidenceNames)
wipConfidences = pd.DataFrame(results['wip_confidence'].to_list(), 
                                    columns=wipConfidenceNames)

# select sub-dataframes to plot
confidences = pd.concat([nsga3Confidences, customConfidences, anchorsConfidences, wipConfidences], axis=1)
confidences = confidences[list(sum(zip(nsga3Confidences.columns, customConfidences.columns, anchorsConfidences.columns, wipConfidences.columns), ()))]
scores = results[["nsga3_score", "custom_score", "anchors_score", "wip_score"]]
times = results[["nsga3_time", "custom_time", "anchors_time", "wip_time"]]

# plots
plotPath = pathToResults + 'plots/'
if not os.path.exists(plotPath):
    os.makedirs(plotPath)

personalizedBoxPlot(confidences, "Confidences comparison", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scores, "Score comparison", path=plotPath)
personalizedBoxPlot(times, "Execution time comparison", path=plotPath, seconds=True, legendInside=True, logscale=True)

# predicted successful adaptations
nsga3PredictedSuccessful = (confidences[nsga3ConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessful = (confidences[customConfidenceNames] > targetConfidence).all(axis=1)
anchorsPredictedSuccessful = (confidences[anchorsConfidenceNames] > targetConfidence).all(axis=1)
wipPredictedSuccessful = (confidences[wipConfidenceNames] > targetConfidence).all(axis=1)

personalizedBoxPlot(confidences[nsga3PredictedSuccessful], "Confidences comparison on NSGA-III predicted success", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scores[nsga3PredictedSuccessful], "Score comparison on NSGA-III predicted success", path=plotPath)
personalizedBoxPlot(times[nsga3PredictedSuccessful], "Execution time comparison on NSGA-III predicted success", path=plotPath, seconds=True, legendInside=True, logscale=True)

print("NSGA-III predicted success rate: " + "{:.2%}".format(nsga3PredictedSuccessful.sum() / nsga3PredictedSuccessful.shape[0]))
print(str(nsga3Confidences.mean()) + "\n")
print("XDA predicted success rate:  " + "{:.2%}".format(customPredictedSuccessful.sum() / customPredictedSuccessful.shape[0]))
print(str(customConfidences.mean()) + "\n")
print("Anchors predicted success rate: " + "{:.2%}".format(anchorsPredictedSuccessful.sum() / anchorsPredictedSuccessful.shape[0]))
print(str(anchorsConfidences.mean()) + "\n")
print("WIP predicted success rate: " + "{:.2%}".format(wipPredictedSuccessful.sum() / wipPredictedSuccessful.shape[0]))
print(str(wipConfidences.mean()) + "\n")

print("NSGA-III mean probas of predicted success: \n" + str(nsga3Confidences[nsga3PredictedSuccessful].mean()) + '\n')
print("XDA mean probas of predicted success: \n" + str(customConfidences[customPredictedSuccessful].mean()) + '\n')
print("Anchors mean probas of predicted success: \n" + str(anchorsConfidences[anchorsPredictedSuccessful].mean()) + '\n')
print("WIP mean probas of predicted success: \n" + str(wipConfidences[wipPredictedSuccessful].mean()) + '\n')

# predicted successful adaptations
nsga3Successful = outcomes[nsga3OutcomeNames].all(axis=1)
customSuccessful = outcomes[customOutcomeNames].all(axis=1)
anchorsSuccessful = outcomes[anchorsOutcomeNames].all(axis=1)
wipSuccessful = outcomes[wipOutcomeNames].all(axis=1)

nsga3SuccessRate = nsga3Successful.mean()
customSuccessRate = customSuccessful.mean()
anchorsSuccessRate = anchorsSuccessful.mean()
wipSuccessRate = wipSuccessful.mean()

# outcomes analysis
print("NSGA-III success rate: " + "{:.2%}".format(nsga3SuccessRate))
print(str(outcomes[nsga3OutcomeNames].mean()) + "\n")
print("XDA success rate:  " + "{:.2%}".format(customSuccessRate))
print(str(outcomes[customOutcomeNames].mean()) + "\n")
print("Anchors success rate: " + "{:.2%}".format(anchorsSuccessRate))
print(str(outcomes[anchorsOutcomeNames].mean()) + "\n")
print("WIP success rate: " + "{:.2%}".format(wipSuccessRate))
print(str(outcomes[wipOutcomeNames].mean()) + "\n")

successRateIndividual = pd.concat([
    outcomes[nsga3OutcomeNames].rename(columns=dict(zip(nsga3OutcomeNames, reqsNamesInGraphs))).mean(),
    outcomes[customOutcomeNames].rename(columns=dict(zip(customOutcomeNames, reqsNamesInGraphs))).mean(),
    outcomes[anchorsOutcomeNames].rename(columns=dict(zip(anchorsOutcomeNames, reqsNamesInGraphs))).mean(),
    outcomes[wipOutcomeNames].rename(columns=dict(zip(wipOutcomeNames, reqsNamesInGraphs))).mean()
], axis=1)

successRateIndividual.columns = ['NSGA-III', 'XDA', 'Anchors', 'WIP']
personalizedBarChart(successRateIndividual, "Success Rate Individual Reqs", plotPath)

successRate = pd.DataFrame([[nsga3SuccessRate, customSuccessRate, anchorsSuccessRate, wipSuccessRate]],
                           columns=["NSGA-III", "XDA", "Anchors", "WIP"])
personalizedBarChart(successRate, "Success Rate", plotPath)

successRateOfPredictedSuccess = pd.DataFrame([[
    outcomes[nsga3OutcomeNames][nsga3PredictedSuccessful].all(axis=1).mean(),
    outcomes[customOutcomeNames][customPredictedSuccessful].all(axis=1).mean(),
    outcomes[anchorsOutcomeNames][anchorsPredictedSuccessful].all(axis=1).mean(),
    outcomes[wipOutcomeNames][wipPredictedSuccessful].all(axis=1).mean()
]],
columns=["NSGA-III", "XDA", "Anchors", "WIP"])
personalizedBarChart(successRateOfPredictedSuccess, "Success Rate of Predicted Success", plotPath)


iterations_per_sample = results["iterations_anchors"]
preds_anch = results["anchors_confidence"]

df_iterations = pd.DataFrame()
df_iterations["iterations_anchors"] = iterations_per_sample
df_iterations["anchors_confidence"] = preds_anch


df_it = pd.DataFrame(df_iterations)

# # Convert list column to numpy arrays for easier computation
# df_it['anchors_confidence'] = df_it['anchors_confidence'].apply(np.array)

# # Group by 'iterations_anchors' and compute mean vector
# grouped = df_it.groupby('iterations_anchors')['anchors_confidence'].apply(lambda x: np.mean(np.stack(x), axis=0))

# # Convert the result to a dataframe where each column is a unique iteration
# final_df = pd.DataFrame(grouped.tolist(), index=grouped.index).T
# final_df.columns = [f'Iter {col}' for col in final_df.columns]
# print(final_df.head())

def plot_iterations(results, iterations_col, confidence_col, name):
    
    df = pd.DataFrame({
        "iterations": results[iterations_col],
        "confidence": results[confidence_col]
    })

    df["confidence"] = df["confidence"].apply(np.array)

    grouped = df.groupby("iterations")["confidence"].apply(
        lambda x: np.mean(np.stack(x), axis=0)
    )

    final_df = pd.DataFrame({k: v for k, v in grouped.items()}).T[0]

    print(final_df.head())
    personalizedBarChart(final_df, name, plotPath)

plot_iterations(results, "iterations_anchors", "anchors_confidence",
                "Anchors Predicted Success w.r.t. it.")

plot_iterations(results, "iterations_wip", "wip_confidence",
                "WIP Predicted Success w.r.t. it.")

