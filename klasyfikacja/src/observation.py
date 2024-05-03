class Observation:
    def __init__(
            self,
            name: str = "",
            all: int = 0,
            TP: int = 0,
            TN: int = 0,
            FP: int = 0,
            FN: int = 0) -> None:

        self.name = name
        self.all = all
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN

    def update(self) -> None:
        self.all = self.TP + self.TN + self.FP + self.FN

    def accuracy(self) -> float:
        accuracy = (self.TP + self.TN) / self.all
        return accuracy

    def precision(self) -> float:
        if self.accuracy() == 0.0:
            return 0.0
        if self.TP == 0:
            return 0.0
        precision = self.TP / (self.TP + self.FP)
        return precision

    def recall(self) -> float:
        if self.accuracy() == 0.0:
            return 0.0
        if self.TP == 0:
            return 0.0
        recall = self.TP / (self.TP + self.FN)
        return recall

    def F1(self) -> float:
        if self.accuracy() == 0.0:
            return 0.0
        if self.TP == 0:
            return 0.0
        p = self.precision()
        r = self.recall()
        F1 = 2 * p * r / (p + r)
        return F1
