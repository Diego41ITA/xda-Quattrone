import numpy as np
import re
import itertools
from math import inf
import os
import explainability_techniques.PDP as pdp
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
import csv
from anchor import utils
from anchor import anchor_tabular
import pandas as pd
from multiprocessing import Pool, cpu_count

from util import vecPredictProba

from typing import Optional, Tuple

import itertools
import time

class AnchorsPlanner:
    """
    A planner that generates, analyzes, and filters *anchor-based explanations* 
    for a set of requirement classifiers applied to a training dataset.

    This class integrates the Anchor Tabular explainer framework with 
    a collection of requirement classifiers to:
      - Train explainers for each requirement
      - Generate anchor explanations for instances satisfying all requirements
      - Combine, intersect, and filter anchors according to statistical thresholds
      - Identify controllable and observable features, and assess their influence
      - Perform adaptation using the polytopes obtained with anchors

    The resulting processed anchors can later be used for reasoning about 
    feature controllability, model confidence, and interpretability in 
    multi-requirement decision systems.

    Attributes
    ----------
    reqClassifiers : list
        List of trained binary classifiers (e.g., scikit-learn models) corresponding 
        to each requirement. Each classifier must implement a `predict()` method.
    reqNames : list of str
        Names of the requirements (one per classifier).
    feature_number : int
        Total number of features in the dataset.
    feature_names : list of str
        Names of all features in the dataset, ordered by index.
    anchorsConfidence : float
        Confidence threshold used when generating anchor explanations.
    controllableFeatureIndices : np.ndarray
        Indices of features considered controllable (modifiable by an agent).
    controllableFeaturesNames : list of str
        Names of controllable features.
    observableFeatureIndices : list of int
        Indices of features that are observable but not controllable.
    observableFeaturesNames : list of str
        Names of observable features.
    controllableFeatureDomains : np.ndarray
        Array specifying the domain (range) of possible values for each controllable feature.
    explanations : list of dict
        Final filtered list of anchor explanations, one per positively classified sample,
        reordered to include all features (with missing anchors filled as unbounded).
    """
    def __init__(self, training_dataset, reqClassifiers, reqNames, 
                 anchorsConfidence, feature_number, feature_names,
                 controllableFeaturesNames, X, plotsPath,
                 controllableFeatureIndices, controllableFeatureDomains,
                 optimizationDirections, optimizationScoreFunction, delta, targetConfidence):
        """
        Initializes the AnchorsPlanner by training requirement explainers, 
        generating anchor-based explanations, and filtering them by confidence.

        Parameters
        ----------
        training_dataset : str
            Path to the CSV file containing the training data. The dataset is expected 
            to include labels for each requirement that indicate whether it is satisfied.
        reqClassifiers : list
            List of trained binary classifiers (one per requirement). Each classifier 
            should expose a `predict()` method compatible with NumPy arrays.
        reqNames : list of str
            List of requirement names, corresponding one-to-one with `reqClassifiers`.
        anchorsConfidence : float
            Confidence level for the anchor explanations (e.g., 0.95).
        feature_number : int
            Number of input features in the dataset.
        feature_names : list of str
            Names of all features (ordered by index).
        controllableFeatureIndices : list of int
            Indices of the features that are controllable (modifiable by the planner).
        controllableFeatureDomains : np.ndarray of shape (n_controllable, 2)
            Domain boundaries for each controllable feature, typically in the format 
            ``[[min_0, max_0], [min_1, max_1], ...]``.

        Notes
        -----
        The initialization procedure performs the following key steps:

        1. Loads the dataset and initializes an `AnchorTabularExplainer` for each requirement.
        2. Compares classifier predictions with ground-truth labels to compute:
            - True positives and negatives
            - False positives and negatives
            - Misclassified samples
        3. Generates anchor explanations for all samples that satisfy all requirements.
        4. Intersects anchor rules across requirements and converts them to numeric intervals.
        5. Reorders the anchors to include all features, filling missing ones with 
           unbounded intervals ``(-inf, inf)``.
        6. Samples random points within anchor-defined intervals to estimate 
           requirement satisfaction probabilities.
        7. Removes anchors whose average predicted confidence is below 0.5.

        At the end of initialization, the `explanations` attribute contains the 
        final filtered anchors corresponding to high-confidence, multi-requirement 
        samples in the training data.

        Examples
        --------
        >>> planner = AnchorsPlanner(
        ...     training_dataset="data/train.csv",
        ...     reqClassifiers=[clf1, clf2],
        ...     reqNames=["SafetyReq", "PerformanceReq"],
        ...     anchorsConfidence=0.95,
        ...     feature_number=5,
        ...     feature_names=["speed", "torque", "temp", "pressure", "humidity"],
        ...     controllableFeatureIndices=[0, 1],
        ...     controllableFeatureDomains=np.array([[0, 100], [0, 50]])
        ... )
        >>> len(planner.explanations)
        12
        """

        # Set random seed for reproducibility
        np.random.seed(42)
        
        self.reqClassifiers = reqClassifiers
        self.reqNames = reqNames
        self.feature_number = feature_number
        self.feature_names = feature_names
        self.anchorsConfidence = anchorsConfidence
        self.controllableFeatureIndices = np.array(controllableFeatureIndices)
        self.controllableFeaturesNames = controllableFeaturesNames

        self.observableFeatureIndices = [i for i in range(feature_number) if i not in controllableFeatureIndices]
        self.observableFeaturesNames = [feature_names[i] for i in self.observableFeatureIndices]
        self.controllableFeatureDomains = controllableFeatureDomains

        self.optimizationDirections = optimizationDirections
        self.optimizationScoreFunction = optimizationScoreFunction
        self.delta = delta
        self.targetConfidence = targetConfidence

        self.merged_polytopes = {}


        # train a k nearest neighbors classifier only used to find the neighbors of a sample in the dataset
        knn = KNeighborsClassifier()
        knn.fit(X.values, np.zeros((X.shape[0],)))
        self.knn = knn

        datasets = []
        features_to_use = [i for i in range(feature_number)]

        # make pdps
        self.pdps = {}
        for i, feature in enumerate(controllableFeaturesNames):
            self.pdps[i] = []
            for j, reqClassifier in enumerate(reqClassifiers):
                path = None
                if plotsPath is not None:
                    path = plotsPath + "/req_" + str(j)
                    if not os.path.exists(path):
                        os.makedirs(path)
                self.pdps[i].append(pdp.partialDependencePlot(reqClassifier, X, [feature], "both", path + "/" + feature + ".png"))

        # make summary pdps
        self.summaryPdps = []
        for i, feature in enumerate(controllableFeaturesNames):
            path = None
            if plotsPath is not None:
                path = plotsPath + "/summary"
                if not os.path.exists(path):
                    os.makedirs(path)
            self.summaryPdps.append(pdp.multiplyPdps(self.pdps[i], path + "/" + feature + ".png"))

        #Creates one dataset for each requirement
        for i,r in enumerate(reqNames):
            datasets.append(\
                utils.load_csv_dataset(\
                    training_dataset, feature_number+i,\
                    features_to_use=features_to_use,\
                    categorical_features=None))

        self.explainer = []
        req_number = len(reqNames)

        for i in range(req_number):
            #initialize the explainer
            self.explainer.append(anchor_tabular.AnchorTabularExplainer(
                datasets[i].class_names, #it maps the 0 and 1 in the dataset's requirements to the class names
                datasets[i].feature_names,
                datasets[i].train,
                datasets[i].categorical_names))
            
        n_samples = datasets[0].train.shape[0]
        K = 2 
        pred_matrix = np.zeros((n_samples, req_number), dtype=int)
        for i in range(req_number):
            pred_matrix[:, i] = reqClassifiers[i].predict(datasets[i].train)

        selected_samples = np.where(pred_matrix.sum(axis=1) >= K)[0]

        explanations = []
        sample_to_rules = {}

        print("Starting anchor generation for", len(selected_samples), "samples satisfying at least", K, "requirements.")
        start_time = time.time()
        for p_sample in selected_samples:
            intersected_exp = {}
            textual_rules = []

            for i in range(req_number):
                #get the sample
                sample = datasets[i].train[p_sample]
                #explain the sample
                exp = self.explainer[i].explain_instance(sample, self.reqClassifiers[i].predict, threshold=0.95)
                #get the textual explanation
                exp = exp.names()
                textual_rules.extend(exp)
                #transform the textual explanations in an interval
                for boundings in exp:
                    quoted, rest = self.__get_anchor(boundings)            
                    if(quoted not in intersected_exp):
                        intersected_exp[quoted] = self.__parse_range(rest)
                    else:
                        intersected_exp[quoted] = self.__intersect(intersected_exp[quoted], self.__parse_range(rest))

            #prepare the data structure
            explanations.append(intersected_exp)
            sample_to_rules[p_sample] = textual_rules

        time_taken = time.time() - start_time
        print("Anchor generation completed in %.2f seconds." % time_taken)

        missing = 0
        explanations_reordered = []
        for exp in explanations:
            exp_reordered = {}
            for k in feature_names:
                if k in exp:
                    exp_reordered[k] = exp[k]
                else:
                    exp_reordered[k] = (-inf, inf, False, False)
                    index = explanations.index(exp)
                    missing = 1
            if missing:
                missing = 0
            explanations_reordered.append(exp_reordered)
            self.explanations = explanations_reordered
        
        number_of_random_points = 10
        number_of_explanations = len(explanations_reordered)
        req_confidences_sum = np.zeros((1, 4))  # Initialize the sum of probabilities for each requirement
        coefficients_to_remove = []
        for n in range(number_of_explanations):
            single_anchor_mean = np.zeros((1, 4))
            single_anchor = explanations[n]
            
            for i in range(number_of_random_points):
                random_sample = np.zeros((1, feature_number))   
                boundary_sample_a = np.zeros((1, feature_number))   
                boundary_sample_b = np.zeros((1, feature_number))   

                for k in single_anchor:
                    a = max(single_anchor[k][0], 0)
                    b = min(single_anchor[k][1], 100)
                    rand = np.random.uniform(a, b)
                    random_sample[0, feature_names.index(k)] = rand
                    boundary_sample_a[0, feature_names.index(k)] = a
                    boundary_sample_b[0, feature_names.index(k)] = b

                probs = vecPredictProba(reqClassifiers, random_sample)
                req_confidences_sum += probs
                single_anchor_mean += probs

            probs_boundary_a = vecPredictProba(reqClassifiers, boundary_sample_a)
            single_anchor_mean += probs_boundary_a
            probs_boundary_b = vecPredictProba(reqClassifiers, boundary_sample_b)
            single_anchor_mean += probs_boundary_b

            single_anchor_mean /= (number_of_random_points + 2)
            if(np.any(single_anchor_mean < 0.5)):
                coefficients_to_remove.append(n)

        req_confidences_sum /= number_of_explanations * number_of_random_points

        # Remove anchors with average probabilities below 0.5
        explanations_removed_negatives = [explanations[i] for i in range(len(explanations)) if i not in coefficients_to_remove]
        explanations = explanations_removed_negatives

        csv_data = []

        for sample_index in selected_samples:
            row = {
                "sample_index": sample_index,
                "rules": "; ".join(sample_to_rules.get(sample_index, [])),
                "n_satisfied_reqs": int(pred_matrix[sample_index].sum())
            }

            # (opzionale ma consigliato)
            for i, req in enumerate(reqNames):
                row[f"pred_{req}"] = int(pred_matrix[sample_index, i])
                row[f"real_{req}"] = int(datasets[i].labels_train[sample_index])

            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        df.to_csv("anchors_rules.csv", index=False)
        print("File anchors_rules.csv salvato correttamente!")

        csv_file = "anchors_explanations.csv"
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)

            # Intestazione
            writer.writerow(["n_satisfied_reqs", "satisfied_reqs"] + feature_names)

            for idx, exp in zip(selected_samples, self.explanations):
                n_satisfied = int(pred_matrix[idx].sum())
                satisfied = [reqNames[i] for i, val in enumerate(pred_matrix[idx]) if val==1]

                row = [idx, n_satisfied, ";".join(satisfied)]
                for feat in feature_names:
                    row.append(str(exp[feat]))  # salva l'intervallo come stringa

                writer.writerow(row)

        print(f"CSV salvato correttamente con {len(selected_samples)} sample!")

    def __get_anchor(self, a)-> tuple:
        """
        Function to separate the name of the feature from the ranges.

        Parameters
        ----------
        a : str
            The string containing the anchor.
        Returns
        -------
        anchor : str
            The anchor.
        rest : str
            The rest of the string.
        """
        quoted_part = a.split("'")[1]
        rest = a.replace(f"'{quoted_part}'", '').replace("b", '').strip()

        return quoted_part, rest

    def __parse_range(self, expr: str):
        """
        Function to parse the range of the anchor.

        Parameters
        ----------
        expr : str
            The string containing the range.
        Returns
        -------
        low : float
            The lower bound of the range.
        high : float
            The upper bound of the range.
        li : bool
            True if the lower bound is included, False otherwise.
        ui : bool
            True if the upper bound is included, False otherwise.
        """
        expr = expr.strip().replace(" ", "")
        
        patterns = [
            (r"^=(\-?\d+(\.\d+)?)$", 'equals'),
            (r"^(>=|>)\s*(-?\d+(\.\d+)?)$", 'lower'),
            (r"^(<=|<)\s*(-?\d+(\.\d+)?)$", 'upper'),
            (r"^(-?\d+(\.\d+)?)(<=|<){1,2}(<=|<)(-?\d+(\.\d+)?)$", 'between'),
            (r"^(-?\d+(\.\d+)?)(>=|>){1,2}(>=|>)(-?\d+(\.\d+)?)$", 'reverse_between'),
        ]
        
        for pattern, kind in patterns:
            match = re.match(pattern, expr)
            if match:
                if kind == 'equals':
                    num = float(match.group(1))
                    return (num, num, True, True)
                elif kind == 'lower':
                    op, num = match.group(1), float(match.group(2))
                    return (
                        num,
                        inf,
                        op == '>=',
                        False
                    )
                elif kind == 'upper':
                    op, num = match.group(1), float(match.group(2))
                    return (
                        -inf,
                        num,
                        False,
                        op == '<='
                    )
                elif kind == 'between':
                    low = float(match.group(1))
                    op1 = match.group(3)
                    op2 = match.group(4)
                    high = float(match.group(5))
                    return (
                        low,
                        high,
                        op1 == '<=',
                        op2 == '<='
                    )
                elif kind == 'reverse_between':
                    high = float(match.group(1))
                    op1 = match.group(3)
                    op2 = match.group(4)
                    low = float(match.group(5))
                    return (
                        low,
                        high,
                        op2 == '>=',
                        op1 == '>='
                    )

        raise ValueError(f"Unrecognized format: {expr}")

    def __inside(self, val, interval) -> bool:
        """
        Function to check if a value is inside an interval.
        Parameters
        ----------
        val : float
            The value to check.
        interval : tuple
            The interval to check.
        Returns
        -------
        bool
            True if the value is inside the interval, False otherwise.
        """
        low, high, li, ui = interval
        if li and ui:
            return low <= val <= high
        elif li and not ui:
            return low <= val < high
        elif not li and ui:
            return low < val <= high
        else:
            return low < val < high
        
    def __intersect(self,
        a: Tuple[float, float, bool, bool],
        b: Tuple[float, float, bool, bool]
    ) -> Optional[Tuple[float, float, bool, bool]]:
        
        a_low, a_high, a_li, a_ui = a
        b_low, b_high, b_li, b_ui = b

        # Compute max of lower bounds
        if a_low > b_low:
            low, li = a_low, a_li
        elif a_low < b_low:
            low, li = b_low, b_li
        else:
            low = a_low
            li = a_li and b_li

        # Compute min of upper bounds
        if a_high < b_high:
            high, ui = a_high, a_ui
        elif a_high > b_high:
            high, ui = b_high, b_ui
        else:
            high = a_high
            ui = a_ui and b_ui

        # Check for empty intersection
        if low > high:
            return None
        if low == high and not (li and ui):
            return None

        return (low, high, li, ui)
    
    def min_dist_polytope(self, x, explanations_table, controllable_features, observable_features):
        """
        Computes the minimum squared Euclidean distance from a given point `x` to a set of 
        polytopes defined by feature intervals for controllable and observable features.

        The distance is measured separately for controllable and observable features by 
        calculating how far `x` lies outside the interval bounds of each polytope. If `x` 
        lies inside the interval for a feature, that feature contributes zero to the distance.

        Parameters:
        -----------
        x : array-like, shape (n_features,)
            Input point for which distances to polytopes are calculated.
            The order of features in `x` is assumed to be [controllable features..., observable features...].
        explanations_table : list of dicts
            Each dict maps feature names to intervals representing a polytope.
            Intervals are tuples/lists in the form (lower_bound, upper_bound, ...).
        controllable_features : list of str
            List of controllable feature names whose intervals are checked against the 
            first part of `x`.
        observable_features : list of str
            List of observable feature names whose intervals are checked against the 
            latter part of `x`. Assumed to start after controllable features in `x`.

        Returns:
        --------
        contr_f_dist : numpy.ndarray
            Array of squared distances between `x` and each polytope considering controllable features only.
        obs_f_dist : numpy.ndarray
            Array of squared distances between `x` and each polytope considering observable features only.
        min_dist_controllable : float
            Minimum squared distance among all polytopes for controllable features.
        min_dist_index_controllable : int
            Index of the polytope with the minimum controllable feature distance.
        min_dist_observable : float
            Minimum squared distance among all polytopes for observable features.
        min_dist_index_observable : int
            Index of the polytope with the minimum observable feature distance.

        Description:
        ------------
        - For each polytope (i.e., each explanation in `explanations_table`), this function:
        1. Iterates over controllable features:
            - If `x`'s value for the feature is outside the polytope interval, 
                adds squared distance from the nearest bound.
            - Stops early if the distance for the current polytope exceeds the minimum found so far.
        2. Repeats the same process for observable features, adjusting indices accordingly.
        - After evaluating all polytopes, finds and returns the minimum distances and their indices.
        - If there are polytopes with zero distance for both controllable and observable features,
        the function favors the polytope common to both sets.

        Notes:
        ------
        - Bounds of `-inf` and `inf` in intervals are replaced by 0 and 100 respectively for distance calculation.
        - The observable features are assumed to start at index offset `len(controllable_features)` in `x`.
        """
        min_dist_controllable = np.inf
        min_dist_index_controllable = -1

        min_dist_observable = np.inf
        min_dist_index_observable = -1

        contr_f_dist = np.zeros(len(explanations_table))
        obs_f_dist = np.zeros(len(explanations_table))

        for i in range(len(explanations_table)):
            for j, f_name in enumerate(controllable_features):
                a, b = explanations_table[i][f_name][0], explanations_table[i][f_name][1]
                if a == -inf:
                    a = 0
                if b == inf:
                    b = 100
                if(x[j] < a):
                    d = (a - x[j]) ** 2
                    contr_f_dist[i] += d
                    if(contr_f_dist[i] >= min_dist_controllable):
                        break
                elif(x[j] > b):
                    d = (b - x[j]) ** 2
                    contr_f_dist[i] += d
                    if(contr_f_dist[i] >= min_dist_controllable):
                        break
            if(contr_f_dist[i] < min_dist_controllable):
                min_dist_controllable = contr_f_dist[i]
                min_dist_index_controllable = i


            for j, f_name in enumerate(observable_features): 
                jj = j + 3
                a, b = explanations_table[i][f_name][0], explanations_table[i][f_name][1]
                if a == -inf:
                    a = 0
                if b == inf:
                    b = 100

                if(x[jj] < a):
                    d = (a - x[jj]) ** 2
                    obs_f_dist[i] += d
                    if(obs_f_dist[i] >= min_dist_observable):
                        break
                elif(x[jj] > b):
                    d = (b - x[jj]) ** 2
                    obs_f_dist[i] += d
                    if(obs_f_dist[i] >= min_dist_observable):
                        break
            if(obs_f_dist[i] < min_dist_observable):
                min_dist_observable = obs_f_dist[i]
                min_dist_index_observable = i

        #This adds the case in which there are muptile polytopes with the same distance, choose the one common to both
        mask_controllable = np.where(contr_f_dist == 0)[0]
        mask_observable = np.where(obs_f_dist == 0)[0]
        for index in mask_observable:
            if index in mask_controllable:
                min_dist_index_observable = index
                min_dist_index_controllable = index
                break

        return contr_f_dist, obs_f_dist, min_dist_controllable, min_dist_index_controllable, min_dist_observable, min_dist_index_observable

    def evaluate_sample(self, sample, threshold = 0.8):
        """
        Method:
        1. Computes the minimum distance from the sample to polytopes (controllable and observable).
        2. Determines if the sample lies inside a polytope:
            - If inside observable AND controllable it evaluate directly with the models.
            - If inside observable but outside controllable it adjust controllable features to move inside.
            - Otherwise it move toward closest observable polytope (adjusting controllable features).
        3. If model confidence is below a threshold, refinement is attempted to improve it
        using `findBestAdaptation()`.

        Parameters
        ----------
        sample : array of shape (n_features,)
            Feature vector representing the system state.

        threshold : float, optional (default = 0.8)
            Minimum acceptable confidence level. If predictions fall below this value,
            sample adaptation is attempted.

        Returns
        -------
        sample : array
            Final sample used for evaluation (may be modified to lie inside a polytope).

        confidence : array
            Per-model confidence scores (probability values).

        outputs : array
            Boolean model predictions for the sample.

        n_iter : int
            Number of adaptation iterations performed (0 if no adaptation occurred).

        Notes
        -----
        - Only controllable features are modified; observable ones remain fixed.
        - A sample inside both polytope types is evaluated immediately.
        - Adaptation aims to increase model confidence beyond the threshold.
        """

        contr_f_dist, obs_f_dist, min_dist_controllable, min_dist_index_controllable, min_dist_observable, min_dist_index_observable = self.min_dist_polytope(sample, self.explanations, self.controllableFeaturesNames, self.observableFeaturesNames)

        if obs_f_dist[min_dist_index_observable] == 0:
            if contr_f_dist[min_dist_index_observable] == 0:
                #Evaluate the sample with the model
                outputs = np.zeros(len(self.reqNames))
                output = True
                for r, req in enumerate(self.reqNames):                    
                    #classify the samplsses with the model
                    tmp_output = self.reqClassifiers[r].predict(sample.reshape(1, -1))
                    outputs[r] = tmp_output
                    output = output and bool(tmp_output)
                
                confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))
                min_prob = np.min(confidence)
                n_iter = 0
                if min_prob < threshold:
                    sample, n_iter = self.findBestAdaptation(sample, self.explanations[min_dist_index_observable], self.controllableFeaturesNames)
                confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))

                return sample, confidence, outputs, n_iter  
            else:
                polytope = self.explanations[min_dist_index_observable]

                sample = self.go_inside_CF_given_polytope(sample, polytope, self.controllableFeaturesNames, self.observableFeaturesNames)
                #check is its now inside the polytope
                for i, f_name in enumerate(self.controllableFeaturesNames):
                    inside = self.__inside(sample[i], polytope[f_name])
                    if not inside:
                        inside = False
                for i, f_name in enumerate(self.observableFeaturesNames):
                    inside = self.__inside(sample[i+3], polytope[f_name])
                    if not inside:
                        inside = False

                #Evaluate the sample with the model
                outputs = np.zeros(len(self.reqNames))
                output = True
                for r, req in enumerate(self.reqNames):
                    #classify the samples with the model
                    tmp_output = self.reqClassifiers[r].predict(sample.reshape(1, -1))
                    outputs[r] = tmp_output
                    output = output and bool(tmp_output)
                contr_f_dist, obs_f_dist, min_dist_controllable, min_dist_index_controllable, min_dist_observable, min_dist_index_observable = self.min_dist_polytope(sample, self.explanations, self.controllableFeaturesNames, self.observableFeaturesNames)
                confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))
                min_prob = np.min(confidence) 
                n_iter = 0
                if min_prob < threshold:
                    sample, n_iter = self.findBestAdaptation(sample, self.explanations[min_dist_index_observable], self.controllableFeaturesNames)
                confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))
                return sample, confidence, outputs, n_iter   
        else:
            #Now we want to change the CF to get as close as possible to that polytope
            polytope = self.explanations[min_dist_index_observable]

            sample = self.go_inside_CF_given_polytope(sample, polytope, self.controllableFeaturesNames, self.observableFeaturesNames)

            #check is its now inside the polytope for the CF
            for i, f_name in enumerate(self.controllableFeaturesNames):
                inside = self.__inside(sample[i], polytope[f_name])
                if not inside:
                    inside = False
            for i, f_name in enumerate(self.observableFeaturesNames):
                inside = self.__inside(sample[i+3], polytope[f_name])
                if not inside:
                    inside = False
            

            

            #Evaluate the sample with the model
            outputs = np.zeros(len(self.reqNames))
            output = True
            for r, req in enumerate(self.reqNames):                
                #classify the samplsses with the model
                tmp_output = self.reqClassifiers[r].predict(sample.reshape(1, -1))
                outputs[r] = tmp_output
                output = output and bool(tmp_output)
            confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))
            min_prob = np.min(confidence)
            n_iter = 0
            if min_prob < threshold:
                sample, n_iter = self.findBestAdaptation(sample, self.explanations[min_dist_index_observable], self.controllableFeaturesNames)
            confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))[0]
            return sample, confidence, outputs, n_iter           
            
    def go_inside_CF_given_polytope(self, sample, polytope, controllable_features, observable_features):
        """
            Adjusts the controllable feature values of a sample to ensure it lies within the bounds of a given polytope.

            Parameters:
            -----------
            sample : array, shape (n_controllable_features + n_observable_features,)
                The input sample whose controllable features will be adjusted.
                The order of features should correspond to controllable features followed by observable features.
            polytope : dict
                A dictionary mapping feature names to intervals, representing the polytope.
                Each interval is expected as a tuple or list: (lower_bound, upper_bound, ...).
            controllable_features : list of str
                List of feature names considered controllable. Only these features are adjusted.
            observable_features : list of str
                List of observable feature names (not used in this function but kept for API consistency).

            Returns:
            --------
            sample : array
                The modified sample where controllable feature values are adjusted to fall within 
                the polytope bounds. If a feature value is outside the bounds, it is moved to 
                one unit inside the closest bound.

            Description:
            ------------
            - For each controllable feature:
                - If the feature value in `sample` is less than the lower bound (with -inf replaced by 0),
                set it to one unit above the lower bound.
                - If the feature value is greater than the upper bound (with inf replaced by 100),
                set it to one unit below the upper bound.
                - Otherwise, the feature value remains unchanged.
            - Observable features are not modified in this method.
            """

        for i, f_name in enumerate(controllable_features):
            a, b = polytope[f_name][0], polytope[f_name][1]
            if a == -inf:
                a = 0
            if b == inf:
                b = 100
            if(sample[i] < int(a)):
                sample[i] = int(a)+ np.abs(b-a)/10
            elif(sample[i] > int(b)):
                sample[i] = int(b) - np.abs(b-a)/10
        return sample

    def findBestAdaptation(self, sample, polytope, controllable_features, threshold=0.8, max_iter=1000):
        """
        Optimize controllable features to increase classifier confidence.

        A local search is performed in the space of controllable features:
        - Small positive/negative deltas are applied to each feature.
        - Modifications are only allowed within the polytope feature bounds.
        - The adaptation that yields the highest average confidence over all classifiers
        is kept.

        Parameters
        ----------
        sample : array
            Initial input sample.

        polytope : dict
            Bounds of allowed values per controllable feature.

        controllable_features : list of str
            Features to be modified.

        threshold : float, optional (default = 0.8)
            Confidence value required to stop adaptation early.

        max_iter : int, optional (default = 1000)
            Maximum number of search iterations.

        Returns
        -------
        adapted_sample : array
            Best sample found (may equal initial sample if no improvement occurred).

        n_iter : int
            Number of iterations executed during adaptation.

        Notes
        -----
        - Search deltas are scaled to 5% of the interval width.
        - If confidence stops improving, deltas are increased (exploration).
        - Early stop if no improvement after repeated attempts.
        """

        delta_controllable_features = []
        for i, f_name in enumerate(controllable_features): #define the delta for each interval
            a, b = polytope[f_name][0], polytope[f_name][1]
            if a == -inf:
                a = 0
            if b == inf:
                b = 100
            delta = (b-a) * 0.05
            delta_controllable_features.append(delta)
            
        n_iter = 0
        min_prob = 0
        best_avg_prob = 0
        early_stopping_condition_counter = 0
                
        adapted_sample = sample

        outOfBounds_feature_counter = 0
        
        while n_iter < max_iter and min_prob < threshold:
            current_MAX_avg_prob = 0

            outOfBounds_feature_counter = 0
            curr_min_prob = 0
            
            current_max_adapted = adapted_sample.copy()
            tmp_sample = adapted_sample.copy()

            for coeff in [1, -1]:
                for i, f_name in enumerate(controllable_features):
                    tmp_sample[i] =adapted_sample[i] + coeff*delta_controllable_features[i]
                    n_iter += 1
                    if tmp_sample[i] >= min(polytope[f_name][1], 100) or tmp_sample[i] <= max(0,polytope[f_name][0]): #If we are outside the bounds of the polytope, we need to stop
                        outOfBounds_feature_counter += 1
                    else:
                        probs = vecPredictProba(self.reqClassifiers, tmp_sample.reshape(1, -1))
                        probs = np.minimum(probs, threshold) #upper bounds each probability to the threshold such that positive probability don't weight too much.
                        current_avg_prob = np.mean(probs)

                        if current_avg_prob > current_MAX_avg_prob:
                            current_MAX_avg_prob = current_avg_prob
                            current_max_adapted = tmp_sample.copy()
                            curr_min_prob = np.min(probs)

            if outOfBounds_feature_counter == 2*len(controllable_features):
                outOfBounds_feature_counter = 0
                break
            
            replaced_flag = 1
            if current_MAX_avg_prob >= best_avg_prob:
                best_avg_prob = current_MAX_avg_prob
                min_prob = curr_min_prob
                adapted_sample = current_max_adapted.copy()
                delta_controllable_features[i] = (b-a) * 0.05

            else:
                current_max_adapted = adapted_sample.copy() #Reset the adapted sample to the best sample found so far
                for i, req in enumerate(controllable_features):
                    delta_controllable_features[i] *= 2 # We increase the delta to explore more
                early_stopping_condition_counter += 1
                if early_stopping_condition_counter >= 100:
                    break

        return adapted_sample, n_iter

        