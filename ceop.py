from aif360.metrics import ClassificationMetric
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import (
    CalibratedEqOddsPostprocessing,
)


class CEOP_Handler:
    def __init__(
        self,
        validation_dataset,
        test_dataset,
        validation_predictions,
        priviliged_groups,
        unpriviliged_groups,
        cost_constraint,
        seed,
    ):
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.validation_predictions = validation_predictions
        self.privileged_groups = priviliged_groups
        self.unprivileged_groups = unpriviliged_groups
        self.seed = seed
        self.datasets = {}

        # Spelling error handling
        if not (
            cost_constraint == "fnr"
            or cost_constraint == "fpr"
            or cost_constraint == "weighted"
        ):
            raise ValueError(
                "cost_constraint must be one of 'fnr', 'fpr', or 'weighted'"
            )

        self.ceop = CalibratedEqOddsPostprocessing(
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
            cost_constraint=cost_constraint,
            seed=self.seed,
        )

        self.ceop.fit(self.validation_dataset, self.validation_predictions)

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
        print("Difference in generalized false positive rates between groups")
        print(round(cm.difference(cm.generalized_false_positive_rate), 3))
        print("Difference in generalized false negative rates between groups")
        print(round(cm.difference(cm.generalized_false_negative_rate), 3))

    def __predict(self, predictions):
        return self.ceop.predict(predictions)

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
