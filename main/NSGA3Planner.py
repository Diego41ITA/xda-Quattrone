import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultMultiObjectiveTermination
from util import vecPredictProba


class NSGA3Planner:
    def __init__(self, reqClassifiers, targetConfidence, controllableFeatureIndices, controllableFeatureDomains,
                 optimizationDirections, successScoreFunction, optimizationScoreFunction,
                 controllableFeatureTypes=None):
        self.reqClassifiers = reqClassifiers
        self.targetConfidence = targetConfidence
        self.successScoreFunction = successScoreFunction
        self.optimizationScoreFunction = optimizationScoreFunction

        # create the reference directions to be used for the optimization
        ref_dirs = get_reference_directions("das-dennis", len(controllableFeatureIndices), n_partitions=12)

        # create the algorithm object
        self.algorithm = NSGA3(ref_dirs=ref_dirs)

        self.termination = DefaultMultiObjectiveTermination(
            cvtol=1e-6,
            period=30,
            n_max_gen=1000
        )

        # create problem instance
        self.problem = Adaptation(reqClassifiers, targetConfidence, self.algorithm.pop_size, controllableFeatureIndices,
                                  controllableFeatureDomains, optimizationDirections,
                                  controllableFeatureTypes=controllableFeatureTypes)

    def findAdaptation(self, sample):
        # The optimizer evolves controllable values only. The problem rebuilds
        # full vectors from this sample so observable positions stay untouched.
        self.problem.original_sample = sample

        # execute the optimization
        res = minimize(self.problem,
                       self.algorithm,
                       seed=1,
                       termination=self.termination)

        if res.X is not None:
            adaptations = res.X
        else:
            adaptations = np.array([individual.X for individual in res.pop])

        adaptations = self.problem.build_full_candidates(adaptations)
        optimizationScores = [self.optimizationScoreFunction(a) for a in adaptations]

        if res.X is not None:
            adaptationIndex = np.argmax(optimizationScores)
        else:
            successScores = [self.successScoreFunction(a, self.reqClassifiers, self.targetConfidence) for a in adaptations]
            adaptationIndex = np.argmax(successScores)

        adaptation = adaptations[adaptationIndex]
        confidence = vecPredictProba(self.reqClassifiers, [adaptation])[0]
        score = self.optimizationScoreFunction(adaptation)

        return adaptation, confidence, score


class Adaptation(Problem):
    def __init__(self, models, targetConfidence, popSize, controllableFeatureIndices, controllableFeatureDomains,
                 optimizationDirections, controllableFeatureTypes=None):
        super().__init__(n_var=len(controllableFeatureIndices), n_obj=len(controllableFeatureIndices),
                         n_constr=len(models), xl=controllableFeatureDomains[:, 0], xu=controllableFeatureDomains[:, 1])
        self.models = models
        self.targetConfidence = np.repeat([targetConfidence], popSize, axis=0)
        self.controllableFeatureIndices = controllableFeatureIndices
        self.optimizationDirections = optimizationDirections
        self.controllableFeatureTypes = tuple(
            controllableFeatureTypes or (float,) * len(controllableFeatureIndices)
        )
        self.popSize = popSize
        self._original_sample = None

    @property
    def original_sample(self):
        return self._original_sample

    @original_sample.setter
    def original_sample(self, sample):
        sample = np.asarray(sample)
        if sample.ndim != 1:
            raise ValueError("NSGA-III requires a full one-dimensional sample")
        if max(self.controllableFeatureIndices, default=-1) >= len(sample):
            raise ValueError("Controllable feature index is outside the sample")
        self._original_sample = sample.copy()

    def build_full_candidates(self, controllable_values):
        if self.original_sample is None:
            raise ValueError("Set original_sample before assembling NSGA-III candidates")
        controllable_values = np.atleast_2d(controllable_values)
        candidates = np.repeat(
            self.original_sample[np.newaxis, :], controllable_values.shape[0], axis=0
        )
        candidates[:, self.controllableFeatureIndices] = controllable_values
        for column, value_type in zip(
            self.controllableFeatureIndices, self.controllableFeatureTypes
        ):
            if value_type is int:
                candidates[:, column] = np.rint(candidates[:, column])
        return candidates

    def _evaluate(self, x, out, *args, **kwargs):
        xFull = self.build_full_candidates(x)
        evaluated_values = xFull[:, self.controllableFeatureIndices]
        out["F"] = evaluated_values * -np.asarray(self.optimizationDirections)
        if self.models:
            out["G"] = self.targetConfidence[0] - vecPredictProba(self.models, xFull)
