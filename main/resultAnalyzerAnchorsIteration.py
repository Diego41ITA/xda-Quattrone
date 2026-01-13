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


def personalizedBoxPlot(data, name, columnNames=None, percentage=False, path=None, show=False, seconds=False, legendInside=False, logscale=False):
    columns = data.columns
    nColumns = len(columns)
    print("Columns:", columns)
    print("Data shape:", data.shape)
    fig = plt.figure()  # plt.figure(figsize=(10, 10 * nColumns/2))
    ax1 = fig.add_subplot(111)  # (nColumns, 1, 1)

    # Creating axes instance
    bp = ax1.boxplot([data[col].dropna().values for col in data.columns],
                 patch_artist=True, notch=True, vert=True)


    colors = plt.cm.Spectral(np.linspace(.1, .9, 5))
    # colors = np.append(colors[0::2], colors[1::2], axis=0)
    c = np.copy(colors)
    for i in range(nColumns//5):
        c = np.append(c, colors, axis=0)

    colors = c

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B',
                    linewidth=1.5,
                    linestyle=":")

    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color='#8B008B',
                linewidth=2)

    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color='red',
                   linewidth=3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker='D',
                  color='#e7298a',
                  alpha=0.5)

    if logscale:
        ax1.set_yscale('log')
    # x-axis labels
    if columnNames is not None and len(columnNames) > 0:
        nGroups = len(columnNames)
        groupSize = 5  # you have 5 algorithms: NSGA-III, XDA, Anchors, Anchors (331-660), Anchors (661-1000)
        positions = np.arange(1, nGroups * groupSize + 1)  # 1..12 for 4 groups
        centers = [np.mean(positions[i*groupSize:(i+1)*groupSize]) for i in range(nGroups)]
        ax1.set_xticks(centers)
        ax1.set_xticklabels(columnNames)
    else:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    # y-axis
    if percentage:
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

        if (data.max().max() - data.min().min())/8 < 0.01:
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    if seconds:
        def y_fmt(x, y):
            return str(int(x)) + ' s' if x >= 1 else str(x) + ' s'
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

    # legend
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    if legendInside:
        ax1.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2], bp["boxes"][3], bp["boxes"][4]], ["NSGA-III", "XDA", "Anchors(0-330)", "Anchors (331-660)", "Anchors (661-1000)"],)
    else:
        ax1.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2], bp["boxes"][3], bp["boxes"][4]], ["NSGA-III", "XDA", "Anchors(0-330)", "Anchors (331-660)", "Anchors (661-1000)"],
                   ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1))

    # Adding title
    plt.title(name)

    # Removing top axes and right axes
    # ticks
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    """
    for i in range(int(nColumns/2)):
        i2 = i + int(nColumns/2)
        axn = fig.add_subplot(nColumns, 1, i + 2)
        subset = data[[columns[i], columns[i2]]]
        subset = subset.sort_values(columns[i2])
        subset = subset.reset_index(drop=True)
        # axn.title.set_text(columns[i] + ' | ' + columns[i + int(nColumns/2)])
        subset.plot(ax=axn, color=colors[[i, i2]])
    """

    if path is not None:
        plt.savefig(path + name)

    if show:
        fig.show()
    else:
        plt.clf()


def personalizedBarChart(data, name, path=None, show=False, percentage=False):
    colors = plt.cm.Spectral(np.linspace(.1, .9, 3))
    # colors = np.append(colors[0::2], colors[1::2], axis=0)
    c = np.copy(colors)
    for i in range(len(data.values) // 3):
        c = np.append(c, colors, axis=0)

    colors = c

    ax = data.plot.bar(title=name, color=colors)

    if len(data.index) > 1:
        plt.xticks(rotation=0)
    else:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax.set_ylim(0, 1)
    if percentage:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    for container in ax.containers:
        if percentage:
            values = ['{:.1%}'.format(v) for v in container.datavalues]
        else:
            values = ['{:.2}'.format(v) for v in container.datavalues]
        ax.bar_label(container, values, fontsize=10)

    """
    for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 5,
            f"{rect.get_height() * 100:.1f}%", fontsize=7,
            ha='center', va='bottom')
    """

    if path is not None:
        plt.savefig(path + name)

    if show:
        plt.show()
    else:
        plt.clf()

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
results = readFromCsv(pathToResults + 'results_15000.csv')
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

# build indices arrays
nsga3ConfidenceNames = ['nsga3_confidence_' + req for req in reqs]
nsga3OutcomeNames = ['nsga3_outcome_' + req for req in reqs]
customConfidenceNames = ['custom_confidence_' + req for req in reqs]
customOutcomeNames = ['custom_outcome_' + req for req in reqs]
anchorsConfidenceNames = ['anchors_confidence_' + req for req in reqs]
anchorsOutcomeNames = ['anchors_outcome_' + req for req in reqs]

#outcomes dataframe
outcomes = pd.concat([nsga3Outcomes[reqs], customOutcomes[reqs], anchorsOutcomes[reqs]], axis=1)
#outcomes.columns = np.append(nsga3OutcomeNames, customOutcomeNames, anchorsOutcomeNames)
outcomes.columns = np.array(nsga3OutcomeNames + customOutcomeNames + anchorsOutcomeNames)

outcomes = outcomes[list(sum(zip(nsga3OutcomeNames, customOutcomeNames, anchorsOutcomeNames), ()))]

# decompose arrays columns into single values columns
nsga3Confidences = pd.DataFrame(results['nsga3_confidence'].to_list(),
                                columns=nsga3ConfidenceNames)
customConfidences = pd.DataFrame(results['custom_confidence'].to_list(),
                                 columns=customConfidenceNames)
anchorsConfidences = pd.DataFrame(results['anchors_confidence'].to_list(),
                                  columns=anchorsConfidenceNames)
anchorsIterations = results['iterations_anchors']
anchorsConfidences0_330 = []
anchorsConfidences331_660 = []
anchorsConfidences661_1000 = []
# Create masks for each iteration range
mask_0_330 = anchorsIterations <= 330
mask_331_660 = (anchorsIterations > 330) & (anchorsIterations <= 660)
mask_661_1000 = anchorsIterations > 660

# Use masks to select rows, fill others with NaN
anchorsConfidences0_330 = anchorsConfidences.where(mask_0_330, np.nan)
anchorsConfidences331_660 = anchorsConfidences.where(mask_331_660, np.nan)
anchorsConfidences661_1000 = anchorsConfidences.where(mask_661_1000, np.nan)


# select sub-dataframes to plot
# First, rename the anchors columns to distinguish between different iteration ranges
anchorsConfidences0_330_renamed = anchorsConfidences0_330.copy()
anchorsConfidences331_660_renamed = anchorsConfidences331_660.copy()
anchorsConfidences661_1000_renamed = anchorsConfidences661_1000.copy()

# Rename columns to distinguish iteration ranges
anchorsConfidences0_330_renamed.columns = [col.replace('anchors_confidence_', 'anchors_0_330_confidence_') for col in anchorsConfidences0_330_renamed.columns]
anchorsConfidences331_660_renamed.columns = [col.replace('anchors_confidence_', 'anchors_331_660_confidence_') for col in anchorsConfidences331_660_renamed.columns]
anchorsConfidences661_1000_renamed.columns = [col.replace('anchors_confidence_', 'anchors_661_1000_confidence_') for col in anchorsConfidences661_1000_renamed.columns]

confidences = pd.concat([nsga3Confidences, customConfidences, anchorsConfidences0_330_renamed, anchorsConfidences331_660_renamed, anchorsConfidences661_1000_renamed], axis=1)

# Reorder columns to group by requirement
reordered_columns = []
for req in reqs:
    reordered_columns.extend([
        f'nsga3_confidence_{req}',
        f'custom_confidence_{req}',
        f'anchors_0_330_confidence_{req}',
        f'anchors_331_660_confidence_{req}',
        f'anchors_661_1000_confidence_{req}'
    ])

confidences = confidences[reordered_columns]
anchors_score_0_330 = results["anchors_score"].where(mask_0_330, np.nan).to_frame('anchors_0_330_score')
anchors_score_331_660 = results["anchors_score"].where(mask_331_660, np.nan).to_frame('anchors_331_660_score')
anchors_score_661_1000 = results["anchors_score"].where(mask_661_1000, np.nan).to_frame('anchors_661_1000_score')
scores = pd.concat([results[["nsga3_score", "custom_score"]], anchors_score_0_330, anchors_score_331_660, anchors_score_661_1000], axis=1)
anchorsTimes0_330 = results["anchors_time"].where(mask_0_330, np.nan).to_frame('anchors_0_330_time')
anchorsTimes331_660 = results["anchors_time"].where(mask_331_660, np.nan).to_frame('anchors_331_660_time')
anchorsTimes661_1000 = results["anchors_time"].where(mask_661_1000, np.nan).to_frame('anchors_661_1000_time')
times = pd.concat([results[["nsga3_time", "custom_time"]], anchorsTimes0_330, anchorsTimes331_660, anchorsTimes661_1000], axis=1)

# plots
plotPath = pathToResults + 'plots_15000_4req_0.8/'
if not os.path.exists(plotPath):
    os.makedirs(plotPath)

personalizedBoxPlot(confidences, "Confidences comparison", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scores, "Score comparison", path=plotPath)
personalizedBoxPlot(times, "Execution time comparison", path=plotPath, seconds=True, legendInside=True, logscale=True)

# predicted successful adaptations
nsga3PredictedSuccessful = (confidences[nsga3ConfidenceNames] > targetConfidence).all(axis=1)
customPredictedSuccessful = (confidences[customConfidenceNames] > targetConfidence).all(axis=1)
# Use the original anchorsConfidences DataFrame for this calculation since we need all anchors data together
anchorsPredictedSuccessful = (anchorsConfidences > targetConfidence).all(axis=1)

personalizedBoxPlot(confidences[nsga3PredictedSuccessful], "Confidences comparison on NSGA-III predicted success", reqsNamesInGraphs, path=plotPath, percentage=False)
personalizedBoxPlot(scores[nsga3PredictedSuccessful], "Score comparison on NSGA-III predicted success", path=plotPath)
personalizedBoxPlot(times[nsga3PredictedSuccessful], "Execution time comparison on NSGA-III predicted success", path=plotPath, seconds=True, legendInside=True, logscale=True)

print("NSGA-III predicted success rate: " + "{:.2%}".format(nsga3PredictedSuccessful.sum() / nsga3PredictedSuccessful.shape[0]))
print(str(nsga3Confidences.mean()) + "\n")
print("XDA predicted success rate:  " + "{:.2%}".format(customPredictedSuccessful.sum() / customPredictedSuccessful.shape[0]))
print(str(customConfidences.mean()) + "\n")
print("Anchors predicted success rate: " + "{:.2%}".format(anchorsPredictedSuccessful.sum() / anchorsPredictedSuccessful.shape[0]))
print(str(anchorsConfidences.mean()) + "\n")

print("NSGA-III mean probas of predicted success: \n" + str(nsga3Confidences[nsga3PredictedSuccessful].mean()) + '\n')
print("XDA mean probas of predicted success: \n" + str(customConfidences[customPredictedSuccessful].mean()) + '\n')
print("Anchors mean probas of predicted success: \n" + str(anchorsConfidences[anchorsPredictedSuccessful].mean()) + '\n')

# predicted successful adaptations
nsga3Successful = outcomes[nsga3OutcomeNames].all(axis=1)
customSuccessful = outcomes[customOutcomeNames].all(axis=1)
anchorsSuccessful = outcomes[anchorsOutcomeNames].all(axis=1)

nsga3SuccessRate = nsga3Successful.mean()
customSuccessRate = customSuccessful.mean()
anchorsSuccessRate = anchorsSuccessful.mean()

# outcomes analysis
print("NSGA-III success rate: " + "{:.2%}".format(nsga3SuccessRate))
print(str(outcomes[nsga3OutcomeNames].mean()) + "\n")
print("XDA success rate:  " + "{:.2%}".format(customSuccessRate))
print(str(outcomes[customOutcomeNames].mean()) + "\n")
print("Anchors success rate: " + "{:.2%}".format(anchorsSuccessRate))
print(str(outcomes[anchorsOutcomeNames].mean()) + "\n")

successRateIndividual = pd.concat([outcomes[nsga3OutcomeNames].rename(columns=dict(zip(nsga3OutcomeNames, reqsNamesInGraphs))).mean(),
                                   outcomes[customOutcomeNames].rename(columns=dict(zip(customOutcomeNames, reqsNamesInGraphs))).mean(),
                                   outcomes[anchorsOutcomeNames].rename(columns=dict(zip(anchorsOutcomeNames, reqsNamesInGraphs))).mean()], axis=1)
successRateIndividual.columns = ['NSGA-III', 'XDA', 'Anchors']
personalizedBarChart(successRateIndividual, "Success Rate Individual Reqs", plotPath)

successRate = pd.DataFrame([[nsga3SuccessRate, customSuccessRate, anchorsSuccessRate]], columns=["NSGA-III", "XDA", "Anchors"])
personalizedBarChart(successRate, "Success Rate", plotPath)

successRateOfPredictedSuccess = pd.DataFrame([[outcomes[nsga3OutcomeNames][nsga3PredictedSuccessful].all(axis=1).mean(),
                                               outcomes[customOutcomeNames][customPredictedSuccessful].all(axis=1).mean(),
                                               outcomes[anchorsOutcomeNames][anchorsPredictedSuccessful].all(axis=1).mean()]],
                                             columns=["NSGA-III", "XDA", "Anchors"])
personalizedBarChart(successRateOfPredictedSuccess, "Success Rate of Predicted Success", plotPath)
