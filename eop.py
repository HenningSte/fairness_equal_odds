from aif360.metrics import ClassificationMetric
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing


class EOP_Handler:
    def __init__(
        self,
        validation_dataset,
        test_dataset,
        validation_predictions,
        priviliged_groups,
        unpriviliged_groups,
        seed,
    ):
        """
        Handler class for the aif360 EqOddsPostprocessing algorithm.
        This class is used to store and manage the datasets and predictions that are
        used in the EOP algorithm.
        It also has methods for dataset metrics and performance metrics.
        """
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.validation_predictions = validation_predictions
        self.privileged_groups = priviliged_groups
        self.unprivileged_groups = unpriviliged_groups
        self.seed = seed
        self.datasets = {}

        self.eop = EqOddsPostprocessing(
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
            seed=self.seed,
        )

        self.eop.fit(self.validation_dataset, self.validation_predictions)

    def add_dataset(self, dataset_name, dataset, predictions):
        self.datasets[dataset_name] = {
            "dataset": dataset,
            "original_predictions": predictions,
            "transformed_predictions": self.__predict(predictions),
            "original_performance_metrics": self.__get_performance_metrics(
                dataset, predictions
            ),
            "transformed_performance_metrics": self.__get_performance_metrics(
                dataset, self.__predict(predictions)
            ),
        }

    def get_dataset(self, name):
        return self.datasets[name]

    def print_dataset_metrics(self, name, transformed):
        dataset = self.get_dataset(name)["dataset"]
        if not transformed:
            predictions = self.get_dataset(name)["original_predictions"]
        else:
            predictions = self.get_dataset(name)["transformed_predictions"]
        cm = ClassificationMetric(
            dataset,
            predictions,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        print("Difference in false positive rates between groups")
        print(round(cm.difference(cm.false_positive_rate), 3))
        print("Difference in false negative rates between groups")
        print(round(cm.difference(cm.false_negative_rate), 3))

    def __predict(self, predictions):
        return self.eop.predict(predictions)

    def __get_performance_metrics(self, dataset, predictions):
        cm = ClassificationMetric(
            dataset,
            predictions,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        accuracy = cm.accuracy()
        fairness = cm.equal_opportunity_difference()
        return accuracy, fairness


if __name__ == "__main__":
    pass
