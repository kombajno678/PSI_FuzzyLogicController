#
# Kod pomocniczy
#

from typing import Union
import numpy as np



class CartForce:
    UNIT_LEFT = -1 # jednostkowe pchnięcie wzóka w lewo [N]
    UNIT_RIGHT = 1 # jednostkowe pchnięcie wózka w prawo [N]
    IDLE_FORCE = 0


class HumanControl(object):
    UserForce = None # type: Union [int, None] # siła, którą użytkownik chce pchnąć wózek
    WantReset = False
    WantPause = False
    WantExit = False


class Keys(object):
    LEFT = 0xFF51
    RIGHT = 0xFF53
    ESCAPE = 0xFF1B
    P = 112
    Q = 113
    R = 114

#######################

class Actions:

    strong_push = 10

    def __init__(self):
        self.actions = {
            'left' : CartForce.UNIT_LEFT * self.strong_push,
            'idle' : 0,
            'right' : CartForce.UNIT_RIGHT * self.strong_push,
        }

    def getAction(self):
        return np.sum(list(self.actions.values()))




