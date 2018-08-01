import numpy as np


class TrueValue:
    alpha1 = np.array(10)

    def set_para(self, value):
        global alpha1
        alpha1 = value
