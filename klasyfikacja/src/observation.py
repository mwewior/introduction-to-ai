class Observation:
    def __init__(
            self,
            all: int = 0,
            TP: int = 0,
            TN: int = 0,
            FP: int = 0,
            FN: int = 0) -> None:

        self.all = all
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN

    def update(self):
        self.all = self.TP + self.TN + self.FP + self.FN
