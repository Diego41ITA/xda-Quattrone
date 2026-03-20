import sys
import os
import glob
import time
import warnings
import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from model.ModelConstructor import constructModel
import explainability_techniques.LIME as lime
from CustomPlanner import CustomPlanner
from NSGA3Planner import NSGA3Planner
from util import vecPredictProba, evaluateAdaptations
from AnchorsPlanner import AnchorsPlanner
from WIP import WIPPlanner
import matplotlib.pyplot as plt

np.random.seed(42)

# success score function (based on the signed distance with respect to the target success probabilities)
def successScore(adaptation, reqClassifiers, targetSuccessProba):
    return np.sum(vecPredictProba(reqClassifiers, [adaptation])[0] - targetSuccessProba)


# provided optimization score function (based on the ideal controllable feature assignment)
def optimizationScore(adaptation):
    return 400 - (100 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3])

# ====================================================================================================== #
# IMPORTANT: everything named as custom in the code refers to the XDA approach                           #
#            everything named as confidence in the code refers to the predicted probabilities of success #
# ====================================================================================================== #


if __name__ == '__main__':
    programStartTime = time.time()

    os.chdir(sys.path[0])

    # suppress all warnings
    warnings.filterwarnings("ignore")

    # evaluate adaptations
    evaluate = True

    ds = pd.read_csv('../datasets/dataset5000.csv')
    featureNames = ["cruise speed",
                    "image resolution",
                    "illuminance",
                    "controls responsiveness",
                    "power",
                    "smoke intensity",
                    "obstacle size",
                    "obstacle distance",
                    "firm obstacle"]
    controllableFeaturesNames = featureNames[0:4]
    externalFeaturesNames = featureNames[4:9]
    controllableFeatureIndices = [0, 1, 2]

    # for simplicity, we consider all the ideal points to be 0 or 100
    # so that we just need to consider ideal directions instead
    # -1 => minimize, 1 => maximize
    optimizationDirections = [1, -1, -1, -1]

    reqs = ["req_0", "req_1", "req_2", "req_3"]

    n_reqs = len(reqs)
    n_neighbors = 10
    n_startingSolutions = 10
    n_controllableFeatures = len(controllableFeaturesNames)

    targetConfidence = np.full((1, n_reqs), 0.8)[0]

    # split the dataset
    X = ds.loc[:, featureNames]
    y = ds.loc[:, reqs]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    trainPath = "../datasets/X_train.csv"

    models = []
    for req in reqs:
        print(Fore.RED + "Requirement: " + req + "\n" + Style.RESET_ALL)

        models.append(constructModel(X_train.values,
                                     X_test.values,
                                     np.ravel(y_train.loc[:, req]),
                                     np.ravel(y_test.loc[:, req])))
        print("=" * 100)

    controllableFeatureDomains = np.repeat([[0, 100]], n_controllableFeatures, 0)

    ds_train = pd.DataFrame(np.hstack((X_train, y_train)), columns=featureNames + reqs)
    trainPath = "../datasets/X_train.csv"
    ds_train.to_csv(trainPath, index=False)

    # initialize planners
    customPlanner = CustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                  controllableFeaturesNames, [0, 1, 2, 3], controllableFeatureDomains,
                                  optimizationDirections, optimizationScore, 1, "../explainability_plots")

    nsga3Planner = NSGA3Planner(models, targetConfidence, [0, 1, 2, 3], controllableFeatureDomains,
                                optimizationDirections, successScore, optimizationScore)
    
    wipPlanner = WIPPlanner(trainPath, models, reqs, 0.95, len(featureNames),featureNames, controllableFeaturesNames, 
                                X_train, "../explainability_plots", controllableFeatureIndices, controllableFeatureDomains
                                    , optimizationDirections, optimizationScore, 1, targetConfidence, build_explanations=False, explanations_csv="anchors_explanations.csv")
    
    # create lime explainer
    limeExplainer = lime.createLimeExplainer(X_train)

    ds_new = pd.read_csv('../datasets/new_dataset.csv')

    # split the dataset
    X = ds_new.loc[:, featureNames]
    y = ds_new.loc[:, reqs]
    X_train_new, X_test_new, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    models_new = []
    for req in reqs:
        print(Fore.RED + "Requirement: " + req + "\n" + Style.RESET_ALL)

        models_new.append(constructModel(X_train_new.values,
                                     X_test_new.values,
                                     np.ravel(y_train.loc[:, req]),
                                     np.ravel(y_test.loc[:, req])))
        print("=" * 100)

    ds_train = pd.DataFrame(np.hstack((X_train_new, y_train)), columns=featureNames + reqs)
    trainPath = "../datasets/X_train_new.csv"
    ds_train.to_csv(trainPath, index=False)
    
    anchorsPlanner = AnchorsPlanner(trainPath, models_new, reqs, 0.95, len(featureNames),featureNames,controllableFeatureIndices, controllableFeatureDomains)

    # metrics
    meanCustomScore = 0
    meanNSGA3Score = 0
    meanWIPScore = 0
    failedAdaptationsCustomNSGA = 0
    failedAdaptationsAnchorsCustom = 0
    failedAdaptationsAnchorsNSGA = 0
    failedAdaptationsAnchorsWip = 0
    meanSpeedupCustomNSGA = 0
    meanSpeedupAnchorsNSGA = 0
    meanSpeedupAnchorsCustom = 0
    meanSpeedupWipAnchors = 0
    meanScoreDiffCustomNSGA = 0
    meanScoreDiffAnchorsCustom = 0
    meanScoreDiffAnchorsNSGA = 0
    meanScoreDiffAnchorsWip = 0

    # adaptations
    results = []
    customDataset = []
    nsga3Dataset = []
    wipDataset = []
    anchorsDataset = []

    path = "../explainability_plots/adaptations"
    if not os.path.exists(path):
        os.makedirs(path)

    files = glob.glob(path + "/*")
    for f in files:
        os.remove(f)

    testNum = 200
    for k in range(1, testNum + 1):
        rowIndex = k - 1
        row = X_test.iloc[rowIndex, :].to_numpy()

        print(Fore.BLUE + "Test " + str(k) + ":" + Style.RESET_ALL)
        print("Row " + str(rowIndex) + ":\n" + str(row))
        print("-" * 100)

        # anchors algorithm
        startTime = time.time()
        customAdaptation_anchors, customConfidence_anchors, _, n_iter_anchors = anchorsPlanner.evaluate_sample(row)
        endTime = time.time()
        
        anchorsTime = endTime - startTime

        if customAdaptation_anchors is not None:
            #keep the features values between 0 and 100
            for ad in range(len(customAdaptation_anchors)):
                ca = customAdaptation_anchors[ad]
                if ca>100:
                    customAdaptation_anchors[ad] = 100
                elif ca<0:
                    customAdaptation_anchors[ad] = 0
            
            customScore_anchors = optimizationScore(customAdaptation_anchors) if customAdaptation_anchors is not None else None
                    
            for i, req in enumerate(reqs):
                lime.saveExplanation(lime.explain(limeExplainer, models_new[i], customAdaptation_anchors),
                                     path + "/" + str(k) + "_" + req + "_final")
                plt.close('all')

            print("Best adaptation Anchors:                 " + str(customAdaptation_anchors[0:n_controllableFeatures]))
            print("Model confidence:                " + str(customConfidence_anchors))
            print("Adaptation score:                " + str(customScore_anchors) + " / 400")
            print("Number of iterations:            " + str(n_iter_anchors))
        else:
            print("No adaptation found")
            customScore_anchors = None

        print("Anchors algorithm execution time: " + str(anchorsTime) + " s")
        print("-" * 100)

        #custom algorithm
        for i, req in enumerate(reqs):
            lime.saveExplanation(lime.explain(limeExplainer, models[i], row),
                                 path + "/" + str(k) + "_" + req + "_starting")
            plt.close('all')

        startTime = time.time()
        customAdaptation, customConfidence, customScore = customPlanner.findAdaptation(row)
        endTime = time.time()
        customTime = endTime - startTime

        if customAdaptation is not None:
            for i, req in enumerate(reqs):
                lime.saveExplanation(lime.explain(limeExplainer, models[i], customAdaptation),
                                     path + "/" + str(k) + "_" + req + "_final")
                plt.close('all')

            print("Best adaptation:                 " + str(customAdaptation[0:n_controllableFeatures]))
            print("Model confidence:                " + str(customConfidence))
            print("Adaptation score:                " + str(customScore) + " / 400")
        else:
            print("No adaptation found")
            customScore = None

        print("Custom algorithm execution time: " + str(customTime) + " s")
        print("-" * 100)

        # wip algorithm
        startTime = time.time()
        customAdaptation_wip, customConfidence_wip, _, n_iter_wip = wipPlanner.evaluate_sample(row)
        endTime = time.time()
        
        wipTime = endTime - startTime

        if customAdaptation_wip is not None:
            #keep the features values between 0 and 100
            for ad in range(len(customAdaptation_wip)):
                ca = customAdaptation_wip[ad]
                if ca>100:
                    customAdaptation_wip[ad] = 100
                elif ca<0:
                    customAdaptation_wip[ad] = 0
            
            customScore_wip = optimizationScore(customAdaptation_wip) if customAdaptation_wip is not None else None
                    
            for i, req in enumerate(reqs):
                lime.saveExplanation(lime.explain(limeExplainer, models[i], customAdaptation_wip),
                                     path + "/" + str(k) + "_" + req + "_final")
                plt.close('all')

            print("Best adaptation WIP:                 " + str(customAdaptation_wip[0:n_controllableFeatures]))
            print("Model confidence:                " + str(customConfidence_wip))
            print("Adaptation score:                " + str(customScore_wip) + " / 400")
            print("Number of iterations:            " + str(n_iter_wip))
        else:
            print("No adaptation found")
            customScore_wip = None

        print("WIP algorithm execution time: " + str(wipTime) + " s")
        print("-" * 100)

        # genetic algorithm
        externalFeatures = row[n_controllableFeatures:]

        startTime = time.time()
        nsga3Adaptation, nsga3Confidence, nsga3Score = nsga3Planner.findAdaptation(externalFeatures)
        endTime = time.time()
        nsga3Time = endTime - startTime

        print("Best NSGA3 adaptation:           " + str(nsga3Adaptation[:n_controllableFeatures]))
        print("Model confidence:                " + str(nsga3Confidence))
        print("Adaptation score:                " + str(nsga3Score) + " / 400")
        print("NSGA3 execution time:            " + str(nsga3Time) + " s")

        print("-" * 100)

        scoreDiffCustomNSGA = None
        scoreImprovementCustomNSGA = None
        
        scoreDiffAnchorsCustom = None
        scoreImprovementAnchorsCustom = None

        scoreDiffAnchorsNSGA = None
        scoreImprovementAnchorsNSGA = None

        scoreDiffAnchorsWip = None
        scoreImprovementAnchorsWip = None

        epsilon = 1e-10

        if anchorsTime == 0:
            anchorsTime = epsilon
        
        speedupCustomNSGA = nsga3Time / customTime #speedup of Custom wrt NSGA3
        speedupAnchorsNSGA = nsga3Time / anchorsTime #speedup of Anchors wrt NSGA3
        speedupAnchorsCustom = customTime / anchorsTime #speedup of Anchors wrt Custom
        speedupWipAnchors = wipTime / anchorsTime #speedup of WIP wrt Anchors

        meanSpeedupCustomNSGA = (meanSpeedupCustomNSGA * (k - 1) + speedupCustomNSGA) / k
        meanSpeedupAnchorsNSGA = (meanSpeedupAnchorsNSGA * (k - 1) + speedupAnchorsNSGA) / k
        meanSpeedupAnchorsCustom = (meanSpeedupAnchorsCustom * (k - 1) + speedupAnchorsCustom) / k
        meanSpeedupWipAnchors = (meanSpeedupWipAnchors * (k - 1) + speedupWipAnchors) / k 
        print(Fore.CYAN + "Speed-up anchors-NSGA3: " + " " * 14 + str(speedupAnchorsNSGA) + "x")
        print(Fore.MAGENTA + "Speed-up anchors-custom: " + " " * 14 + str(speedupAnchorsCustom) + "x")
        print(Fore.GREEN + "Speed-up custom-NSGA3: " + " " * 14 + str(speedupCustomNSGA) + "x")
        print(Fore.YELLOW + "Speed-up WIP-Anchors: " + " " * 17 + str(speedupWipAnchors) + "x")

        if customAdaptation is not None and nsga3Adaptation is not None:
            scoreDiffCustomNSGA = customScore - nsga3Score
            scoreImprovementCustomNSGA = scoreDiffCustomNSGA / nsga3Score
            print(Fore.GREEN + "Score Custom NSGA diff:        " + " " * 5 + str(scoreDiffCustomNSGA))
            print(Fore.GREEN + "Score Custom NSGA improvement: " + " " * 5 + "{:.2%}".format(scoreImprovementCustomNSGA))
        else:
            failedAdaptationsCustomNSGA += 1

        print(Style.RESET_ALL + Fore.GREEN + "Mean speed-up Custom NSGA: " + " " * 9 + str(meanSpeedupCustomNSGA) + "x")
        
        if customAdaptation is not None and customAdaptation_anchors is not None:
            scoreDiffAnchorsCustom = customScore_anchors - customScore
            scoreImprovementAnchorsCustom = scoreDiffAnchorsCustom / customScore
            print(Fore.MAGENTA + "Score diff:        " + " " * 5 + str(scoreDiffAnchorsCustom))
            print(Fore.MAGENTA + "Score improvement: " + " " * 5 + "{:.2%}".format(scoreImprovementAnchorsCustom))
        else:
            failedAdaptationsAnchorsCustom += 1

        print(Style.RESET_ALL + Fore.MAGENTA + "Mean speed-up Anchors Custom: " + " " * 5 + str(meanSpeedupAnchorsCustom) + "x")

        if customAdaptation_anchors is not None and nsga3Adaptation is not None:
            scoreDiffAnchorsNSGA = customScore_anchors - nsga3Score
            scoreImprovementAnchorsNSGA = scoreDiffAnchorsNSGA / nsga3Score
            print(Fore.CYAN + "Score Anchors NSGA diff:        " + " " * 5 + str(scoreDiffAnchorsNSGA))
            print(Fore.CYAN + "Score Anchors NSGA improvement: " + " " * 5 + "{:.2%}".format(scoreImprovementAnchorsNSGA))
        else:
            failedAdaptationsAnchorsNSGA += 1
        
        print(Style.RESET_ALL + Fore.CYAN + "Mean speed-up Anchors NSGA: " + " " * 9 + str(meanSpeedupAnchorsNSGA) + "x")

        if customAdaptation_wip is not None and customAdaptation_anchors is not None:
            scoreDiffAnchorsWip = customScore_anchors - customScore_wip
            scoreImprovementAnchorsWip = scoreDiffAnchorsWip / customScore_wip
            print(Fore.YELLOW + "Score Anchors WIP diff:        " + " " * 5 + str(scoreDiffAnchorsWip))
            print(Fore.YELLOW + "Score Anchors WIP improvement: " + " " * 5 + "{:.2%}".format(scoreImprovementAnchorsWip))
        else:
            failedAdaptationsAnchorsWip += 1

        print(Style.RESET_ALL + Fore.YELLOW + "Mean speed-up Anchors WIP: " + " " * 9 + str(meanSpeedupWipAnchors) + "x")

        if customAdaptation is not None and nsga3Adaptation is not None:
            meanCustomScore = (meanCustomScore * (k - 1 - failedAdaptationsCustomNSGA) + customScore) / (k - failedAdaptationsCustomNSGA)
            meanNSGA3Score = (meanNSGA3Score * (k - 1 - failedAdaptationsCustomNSGA) + nsga3Score) / (k - failedAdaptationsCustomNSGA)
            meanScoreDiffCustomNSGA = (meanScoreDiffCustomNSGA * (k - 1 - failedAdaptationsCustomNSGA) + scoreDiffCustomNSGA) / (k - failedAdaptationsCustomNSGA)
            meanScoreImprovementCustomNSGA = meanScoreDiffCustomNSGA / meanNSGA3Score
            print(Fore.GREEN + "Mean score diff:        " + str(meanScoreDiffCustomNSGA))
            print(Fore.GREEN + "Mean score improvement: " + "{:.2%}".format(meanScoreImprovementCustomNSGA))
        
        if customAdaptation_anchors is not None and customAdaptation is not None:
            meanAnchorsScore = (meanCustomScore * (k - 1 - failedAdaptationsAnchorsCustom) + customScore_anchors) / (k - failedAdaptationsAnchorsCustom)
            meanCustomScore = (meanCustomScore * (k - 1 - failedAdaptationsAnchorsCustom) + customScore) / (k - failedAdaptationsAnchorsCustom)
            meanScoreDiffAnchorsCustom = (meanScoreDiffAnchorsCustom * (k - 1 - failedAdaptationsAnchorsCustom) + scoreDiffAnchorsCustom) / (k - failedAdaptationsAnchorsCustom)
            meanScoreImprovementAnchorsCustom = meanScoreDiffAnchorsCustom / meanCustomScore
            print(Fore.MAGENTA + "Mean anchors custom score diff:        " + str(meanScoreDiffAnchorsCustom))
            print(Fore.MAGENTA + "Mean anchors custom score improvement: " + "{:.2%}".format(meanScoreImprovementAnchorsCustom))
        
        if customAdaptation_anchors is not None and nsga3Adaptation is not None:
            meanAnchorsScore = (meanAnchorsScore * (k - 1 - failedAdaptationsAnchorsNSGA) + customScore_anchors) / (k - failedAdaptationsAnchorsNSGA)
            meanNSGA3Score = (meanNSGA3Score * (k - 1 - failedAdaptationsAnchorsNSGA) + nsga3Score) / (k - failedAdaptationsAnchorsNSGA)
            meanScoreDiffAnchorsNSGA = (meanScoreDiffAnchorsNSGA * (k - 1 - failedAdaptationsAnchorsNSGA) + scoreDiffAnchorsNSGA) / (k - failedAdaptationsAnchorsNSGA)
            meanScoreImprovementAnchorsNSGA = meanScoreDiffAnchorsNSGA / meanNSGA3Score
            print(Fore.CYAN + "Mean anchors NSGA score diff:        " + str(meanScoreDiffAnchorsNSGA))
            print(Fore.CYAN + "Mean anchors NSGA score improvement: " + "{:.2%}".format(meanScoreImprovementAnchorsNSGA))

        if customAdaptation_wip is not None and customAdaptation_anchors is not None:
            meanAnchorsScore = (meanAnchorsScore * (k - 1 - failedAdaptationsAnchorsWip) + customScore_anchors) / (k - failedAdaptationsAnchorsWip)
            meanWIPScore = (meanWIPScore * (k - 1 - failedAdaptationsAnchorsWip) + customScore_wip) / (k - failedAdaptationsAnchorsWip)
            meanScoreDiffAnchorsWip = (meanScoreDiffAnchorsWip * (k - 1 - failedAdaptationsAnchorsWip) + scoreDiffAnchorsWip) / (k - failedAdaptationsAnchorsWip)
            meanScoreImprovementAnchorsWip = meanScoreDiffAnchorsWip / meanWIPScore
            print(Fore.YELLOW + "Mean anchors WIP score diff:        " + str(meanScoreDiffAnchorsWip))
            print(Fore.YELLOW + "Mean anchors WIP score improvement: " + "{:.2%}".format(meanScoreImprovementAnchorsWip))

        print(Style.RESET_ALL + "=" * 100)

        results.append([nsga3Adaptation, customAdaptation, customAdaptation_anchors, customAdaptation_wip,
                        nsga3Confidence, customConfidence, customConfidence_anchors, customConfidence_wip,
                        nsga3Score, customScore, customScore_anchors, customScore_wip,
                        scoreDiffCustomNSGA, scoreDiffAnchorsCustom, scoreDiffAnchorsNSGA, scoreDiffAnchorsWip,
                        scoreImprovementCustomNSGA,scoreImprovementAnchorsCustom, scoreImprovementAnchorsNSGA, scoreImprovementAnchorsWip,
                        nsga3Time, customTime, anchorsTime, wipTime,
                        speedupCustomNSGA, speedupAnchorsNSGA, speedupAnchorsCustom, speedupWipAnchors, n_iter_anchors, n_iter_wip])
        


    results = pd.DataFrame(results, columns=["nsga3_adaptation", "custom_adaptation", "anchors_adaptation", "wip_adaptation",
                                             "nsga3_confidence", "custom_confidence", "anchors_confidence", "wip_confidence",
                                             "nsga3_score", "custom_score", "anchors_score", "wip_score",
                                             "score_diff_custom_NSGA", "score_diff_anchors_custom", "score_diff_anchors_nsga", "score_diff_anchors_wip",
                                             "score_improvement_NSGA_custom[%]", "score_improvement_anchors_custom[%]", "score_improvement_anchors_NSGA[%]", "score_improvement_anchors_WIP[%]",
                                             "nsga3_time", "custom_time", "anchors_time", "wip_time",
                                             "speed-up_custom_NSGA", "speed-up_anchors_NSGA", "speed-up_anchors_custom", "speed-up_WIP_anchors", "iterations_anchors", "iterations_wip"])
    path = "../results"
    if not os.path.exists(path):
        os.makedirs(path)
    results.to_csv(path + "/results_new.csv")

    if evaluate:
        evaluateAdaptations(results, featureNames)

    programEndTime = time.time()
    totalExecutionTime = programEndTime - programStartTime
    print("\nProgram execution time: " + str(totalExecutionTime / 60) + " m")
