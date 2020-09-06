#
# Kod pomocniczy
#

from typing import Union
import numpy as np
import skfuzzy as fuzz



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

class Response:

    force = 25
    defuzze_method = 'centroid'
    print_info = False

    def __init__(self):
        self.actions = {
            'left' : 0,
            'idle' : 0,
            'right' : 0,
        }
       

    def defuzze(self):

        left_range = self.actions['left']
        idle_range = self.actions['idle']
        right_range = self.actions['right']

        force_range = np.arange(-self.force*2, self.force*2+0.1, 0.1)

        force_left  = fuzz.trapmf(force_range, [-self.force*2, -self.force-1, -self.force+1, 0])
        force_idle  = fuzz.trapmf(force_range, [-self.force+1, 0, 0, self.force-1])
        force_right = fuzz.trapmf(force_range, [0, self.force-1, self.force+1, self.force*2])


        """
        4. Dla każdej reguły przeprowadź operację wnioskowania Mamdaniego.
            Operatorem wnioskowania jest min().
            Przykład: Jeżeli lingwistyczna zmienna wyjściowa ForceToApply ma 5 wartości (strong left, light left, idle, light right, strong right)
            to liczba wyrażeń wnioskujących wyniesie 5 - po jednym wywołaniu operatora Mamdaniego dla konkluzji.
            
            W ten sposób wyznaczasz aktywacje poszczególnych wartości lingwistycznej zmiennej wyjściowej.
            Uważaj - aktywacja wartości zmiennej lingwistycznej w konkluzji to nie liczba a zbiór rozmyty.++++
            Ponieważ stosujesz operator min(), to wynikiem będzie "przycięty od góry" zbiór rozmyty. +++
        """
        force_left  = np.fmin(force_left, left_range)
        force_idle  = np.fmin(force_idle, idle_range)
        force_right = np.fmin(force_right, right_range)

        """
        5. Agreguj wszystkie aktywacje dla danej zmiennej wyjściowej.
        """

        aggregated = np.fmax(force_left, np.fmax(force_idle, force_right))

        """
        6. Dokonaj defuzyfikacji (np. całkowanie ważone - centroid).+++
        """

        defuzzed  = fuzz.defuzz(force_range, aggregated, self.defuzze_method)

        """
        7. Czym będzie wyjściowa wartość skalarna?
        """

        if self.print_info :
            print(f"defuzzed = {defuzzed} (method: {self.defuzze_method})")

        return defuzzed






