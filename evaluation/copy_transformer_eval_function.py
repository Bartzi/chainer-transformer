from chainer import reporter, function
from chainer.backends import cuda
from chainer.training.extensions import Evaluator


class CopyTransformerEvaluationFunction:

    def __init__(self, net, device):
        self.net = net
        self.device = device

    def __call__(self, **kwargs):
        data = kwargs.pop('data')
        labels = kwargs.pop('label')

        with cuda.Device(self.device):
            prediction = self.net.predict(data)
            part_accuracy, accuracy = self.calc_accuracy(prediction, labels)

        reporter.report({
            "part_accuracy": part_accuracy,
            "accuracy": accuracy
        })

    def calc_accuracy(self, predictions, labels):
        correct_lines = 0
        correct_parts = 0
        for predicted_item, item in zip(predictions, labels):
            accuracy_result = (predicted_item == item).sum()
            correct_parts += accuracy_result

            if accuracy_result == predictions.shape[1]:
                correct_lines += 1

        return correct_parts / predictions.size, correct_lines / len(predictions)


class CopyTransformerEvaluator(Evaluator):

    def evaluate(self):
        summary = reporter.DictSummary()
        eval_func = self.eval_func or self._targets['main']

        observation = {}
        with reporter.report_scope(observation):
            data = eval_func.net.xp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype='int32')
            eval_func(data=data, label=data)

        summary.add(observation)
        return summary.compute_mean()
