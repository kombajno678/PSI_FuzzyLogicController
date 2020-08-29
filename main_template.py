#
# Podstawy Sztucznej Inteligencji, IIS 2020
# Autor: Tomasz Jaworski
# Opis: Szablon kodu do stabilizacji odwróconego wahadła (patyka) w pozycji pionowej podczas ruchu wózka.
#

import gym # Instalacja: https://github.com/openai/gym
import time
from helper import HumanControl, Keys, CartForce, Actions
import matplotlib.pyplot as plt


import numpy as np
import skfuzzy as fuzz

#
# przygotowanie środowiska
#
control = HumanControl()
env = gym.make('gym_PSI:CartPole-v2')
env.reset()
env.render()


def on_key_press(key: int, mod: int):
    global control
    force = 10
    if key == Keys.LEFT:
        control.UserForce = force * CartForce.UNIT_LEFT # krok w lewo
    if key == Keys.RIGHT:
        control.UserForce = force * CartForce.UNIT_RIGHT # krok w prawo
    if key == Keys.P: # pauza
        control.WantPause = True
    if key == Keys.R: # restart
        control.WantReset = True
    if key == Keys.ESCAPE or key == Keys.Q: # wyjście
        control.WantExit = True

env.unwrapped.viewer.window.on_key_press = on_key_press

#########################################################
# KOD INICJUJĄCY - do wypełnienia
#########################################################


def OR(a, b):
    return np.fmax(a, b)

def AND(a, b):
    return np.fmin(a, b)

#
# pole angle 
#
pole_angle_target = 0;
pole_angle_d = 0.1

pole_angle_range = np.arange(-3, 3, 0.01)

pole_angle_l1 = pole_angle_target - pole_angle_d
pole_angle_r1 = pole_angle_target + pole_angle_d

pole_angle_left     = fuzz.trapmf(pole_angle_range, [-3, -3, pole_angle_l1, pole_angle_target])
pole_angle_vertical  = fuzz.trimf(pole_angle_range, [pole_angle_l1, pole_angle_target, pole_angle_r1])
pole_angle_right    = fuzz.trapmf(pole_angle_range, [pole_angle_target, pole_angle_r1, 3, 3])

#
# cart_position
#
cart_position_target = 0
cart_position_d = 0.35

cart_position_range = np.arange(-5, 5, 0.01)

cart_position_l1 = cart_position_target - cart_position_d
cart_position_r1 = cart_position_target + cart_position_d

cart_position_left    = fuzz.trapmf(cart_position_range, [-5, -5, cart_position_l1, cart_position_target])
cart_position_desired  = fuzz.trimf(cart_position_range, [cart_position_l1, cart_position_target, cart_position_r1])
cart_position_right   = fuzz.trapmf(cart_position_range, [cart_position_target, cart_position_r1, 5, 5])


#
# cart_velocity
#
cart_velocity_target = 0
cart_velocity_d = 1

cart_velocity_range = np.arange(-10, 10, 0.01)

cart_velocity_l1 = cart_velocity_target - cart_velocity_d
cart_velocity_r1 = cart_velocity_target + cart_velocity_d

cart_velocity_left  = fuzz.trapmf(cart_velocity_range, [-10, -10, cart_velocity_l1, cart_velocity_target])
cart_velocity_zero   = fuzz.trimf(cart_velocity_range, [cart_velocity_l1, cart_velocity_target, cart_velocity_r1])
cart_velocity_right = fuzz.trapmf(cart_velocity_range, [cart_velocity_target, cart_velocity_r1, 10, 10])



"""

1. Określ dziedzinę dla każdej zmiennej lingwistycznej. Każda zmienna ma własną dziedzinę.
2. Zdefiniuj funkcje przynależności dla wybranych przez siebie zmiennych lingwistycznych.
3. Wyświetl je, w celach diagnostycznych.

Przykład wyświetlania:
"""
if True:
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

    ax0.plot(pole_angle_range, pole_angle_left, 'b', linewidth=1.5, label='Left')
    ax0.plot(pole_angle_range, pole_angle_vertical, 'k', linewidth=1.5, label='Zero')
    ax0.plot(pole_angle_range, pole_angle_right, 'r', linewidth=1.5, label='Right')
    ax0.set_title('Pole angle')
    ax0.legend()

    ax1.plot(cart_position_range, cart_position_left, 'b', linewidth=1.5, label='Left')
    ax1.plot(cart_position_range, cart_position_desired, 'k', linewidth=1.5, label='Zero')
    ax1.plot(cart_position_range, cart_position_right, 'r', linewidth=1.5, label='Right')
    ax1.set_title('Cart position')
    ax1.legend()
    
    ax2.plot(cart_velocity_range, cart_velocity_left, 'b', linewidth=1.5, label='Left')
    ax2.plot(cart_velocity_range, cart_velocity_zero, 'k', linewidth=1.5, label='Zero')
    ax2.plot(cart_velocity_range, cart_velocity_right, 'r', linewidth=1.5, label='Right')
    ax2.set_title('Cart velocity')
    ax2.legend()

    plt.tight_layout()
    plt.show()
"""
"""
#########################################################
# KONIEC KODU INICJUJĄCEGO
#########################################################


#
# Główna pętla symulacji
#
while not control.WantExit:

    #
    # Wstrzymywanie symulacji:
    # Pierwsze wciśnięcie klawisza 'p' wstrzymuje; drugie wciśnięcie 'p' wznawia symulację.
    #
    if control.WantPause:
        control.WantPause = False
        while not control.WantPause:
            time.sleep(0.1)
            env.render()
        control.WantPause = False

    #
    # Czy użytkownik chce zresetować symulację?
    if control.WantReset:
        control.WantReset = False
        env.reset()


    ###################################################
    # ALGORYTM REGULACJI - do wypełnienia
    ##################################################

    """
    Opis wektora stanu (env.state)
        cart_position   -   Położenie wózka w osi X. Zakres: -2.5 do 2.5. Ppowyżej tych granic wózka znika z pola widzenia.
        cart_velocity   -   Prędkość wózka. Zakres +- Inf, jednak wartości powyżej +-2.0 generują zbyt szybki ruch.
        pole_angle      -   Pozycja kątowa patyka, a<0 to odchylenie w lewo, a>0 odchylenie w prawo. Pozycja kątowa ma
                            charakter bezwzględny - do pozycji wliczane są obroty patyka.
                            Ze względów intuicyjnych zaleca się konwersję na stopnie (+-180).
        tip_velocity    -   Prędkość wierzchołka patyka. Zakres +- Inf. a<0 to ruch przeciwny do wskazówek zegara,
                            podczas gdy a>0 to ruch zgodny z ruchem wskazówek zegara.
                            
    Opis zadajnika akcji (fuzzy_response):
        Jest to wartość siły przykładana w każdej chwili czasowej symulacji, wyrażona w Newtonach.
        Zakładany krok czasowy symulacji to env.tau (20 ms).
        Przyłożenie i utrzymanie stałej siły do wózka spowoduje, że ten będzie przyspieszał do nieskończoności,
        ruchem jednostajnym.
    """

    cart_position, cart_velocity, pole_angle, tip_velocity = env.state # Wartości zmierzone


    if cart_position > 3 or cart_position < -3:
        env.reset()
    if pole_angle > 3 or pole_angle < -3:
        env.reset()


   


    """
    
    1. Przeprowadź etap rozmywania, w którym dla wartości zmierzonych wyznaczone zostaną ich przynależności do poszczególnych
       zmiennych lingwistycznych. Jedno fizyczne wejście (źródło wartości zmierzonych, np. położenie wózka) posiada własną
       zmienną lingwistyczną.
       
       Sprawdź funkcję interp_membership
       
    2. Wyznacza wartości aktywacji reguł rozmytych, wyznaczając stopień ich prawdziwości.
       Przykład reguły:
       JEŻELI kąt patyka jest zerowy ORAZ prędkość wózka jest zerowa TO moc chwilowa jest zerowa
       JEŻELI kąt patyka jest lekko ujemny ORAZ prędkość wózka jest zerowa TO moc chwilowa jest lekko ujemna
       JEŻELI kąt patyka jest średnio ujemny ORAZ prędkość wózka jest lekko ujemna TO moc chwilowa jest średnio ujemna
       JEŻELI kąt patyka jest szybko rosnący w kierunku ujemnym TO moc chwilowa jest mocno ujemna
       .....
       
       Przyjmując, że spójnik LUB (suma rozmyta) to max() a ORAZ/I (iloczyn rozmyty) to min() sprawdź funkcje fmax i fmin.
    
    
    3. Przeprowadź agregację reguł o tej samej konkluzji.
       Jeżeli masz kilka reguł, posiadających tę samą konkluzję (ale różne przesłanki) to poziom aktywacji tych reguł
       należy agregować tak, aby jedna konkluzja miała jeden poziom aktywacji. Skorzystaj z sumy rozmytej.
    
    4. Dla każdej reguły przeprowadź operację wnioskowania Mamdaniego.
       Operatorem wnioskowania jest min().
       Przykład: Jeżeli lingwistyczna zmienna wyjściowa ForceToApply ma 5 wartości (strong left, light left, idle, light right, strong right)
       to liczba wyrażeń wnioskujących wyniesie 5 - po jednym wywołaniu operatora Mamdaniego dla konkluzji.
       
       W ten sposób wyznaczasz aktywacje poszczególnych wartości lingwistycznej zmiennej wyjściowej.
       Uważaj - aktywacja wartości zmiennej lingwistycznej w konkluzji to nie liczba a zbiór rozmyty.
       Ponieważ stosujesz operator min(), to wynikiem będzie "przycięty od góry" zbiór rozmyty. 
       
    5. Agreguj wszystkie aktywacje dla danej zmiennej wyjściowej.
    
    6. Dokonaj defuzyfikacji (np. całkowanie ważone - centroid).
    
    7. Czym będzie wyjściowa wartość skalarna?
    
    """
    fuzzy_response = CartForce.IDLE_FORCE # do zmiennej fuzzy_response zapisz wartość siły, jaką chcesz przyłożyć do wózka.

    cart_position_multiplier = 1
    pole_angle_multiplier = 1

    # pole angle
    is_pole_angle_left =     pole_angle_multiplier * fuzz.interp_membership(pole_angle_range, pole_angle_left,           pole_angle)
    is_pole_angle_vertical = pole_angle_multiplier * fuzz.interp_membership(pole_angle_range, pole_angle_vertical,       pole_angle)
    is_pole_angle_right =    pole_angle_multiplier * fuzz.interp_membership(pole_angle_range, pole_angle_right,          pole_angle)
    #print(
    #    f"pole angle [{is_pole_angle_left:8.4f} {is_pole_angle_vertical:8.4f} {is_pole_angle_right:8.4f}]")

    # cart position
    is_cart_position_left =    cart_position_multiplier * fuzz.interp_membership(cart_position_range, cart_position_left,         cart_position)
    is_cart_position_desired = cart_position_multiplier * fuzz.interp_membership(cart_position_range, cart_position_desired,      cart_position)
    is_cart_position_right =   cart_position_multiplier * fuzz.interp_membership(cart_position_range, cart_position_right,        cart_position)
    #print(
    #    f"cart posit [{is_cart_position_left:8.4f} {is_cart_position_desired:8.4f} {is_cart_position_right:8.4f}]")

    # cart velocity
    is_cart_velocity_left =         fuzz.interp_membership(cart_velocity_range, cart_velocity_left,         cart_velocity)
    is_cart_velocity_zero =         fuzz.interp_membership(cart_velocity_range, cart_velocity_zero,         cart_velocity)
    is_cart_velocity_right =        fuzz.interp_membership(cart_velocity_range, cart_velocity_right,        cart_velocity)
    #print(
    #    f"cart veloc [{is_cart_velocity_left:8.4f} {is_cart_velocity_zero:8.4f} {is_cart_velocity_right:8.4f}]")


    # init actions
    actions = Actions()

    


    # rules for cart position and cart velocity

    """
    #if cart is right of desired position AND cart velocity is not left
    #if cart is in desired poition AND cart velocity is right
    #if pole is left 
    left1 = cart_position_multiplier * np.fmin(is_cart_position_right, 1 - is_cart_velocity_left)
    left2 = cart_position_multiplier * np.fmin(is_cart_position_desired, is_cart_velocity_right)
    left3 = pole_angle_multiplier * is_pole_angle_left
    actions.actions['left'] *= np.fmax(np.fmax(left1, left2), left3)
    """
    #if pole is left AND (cart is right of desired position)
    #if (cart is in desired poition AND cart velocity is right)
    #if pole is left 
    #if pole is vertical AND cart is left
    left1 = AND(is_pole_angle_left, is_cart_position_right)
    left2 = 0#AND(is_cart_position_desired, is_cart_velocity_right)
    left3 = is_pole_angle_left
    left4 = AND(is_pole_angle_vertical, is_cart_position_left)
    

    actions.actions['left'] *= OR(OR(left1, left2), OR(left3, left4))

    """
    #if cart is left of desired position AND cart velocity is not right
    #if cart is in desired poition AND cart velocity is left 
    #if pole is right 
    right1 = cart_position_multiplier * np.fmin(is_cart_position_left, 1 - is_cart_velocity_right)
    right2 = cart_position_multiplier * np.fmin(is_cart_position_desired, is_cart_velocity_left)
    right3 = pole_angle_multiplier * is_pole_angle_right
    actions.actions['right'] *= np.fmax(np.fmax(right1, right2), right3) 
    """

    #if pole is right AND (cart is right of desired position)
    #if cart is in desired poition AND cart velocity is right)
    #if pole is right 
    # if pole is vertical ANd cart is right
    right1 = AND(is_pole_angle_right, is_cart_position_left)
    right2 = 0#AND(is_cart_position_desired, is_cart_velocity_left)
    right3 = is_pole_angle_right
    right4 = AND(is_pole_angle_vertical, is_cart_position_right)

    actions.actions['right'] *= OR(OR(right1, right2), OR(right3, right4)) 


    
    #print(actions.actions)
    fuzzy_response = actions.getAction()
    #print(f"fuzzy_response = {fuzzy_response}")
    #
    # KONIEC algorytmu regulacji
    #########################

    # Jeżeli użytkownik chce przesunąć wózek, to jego polecenie ma wyższy priorytet
    if control.UserForce is not None:
        applied_force = control.UserForce
        control.UserForce = None
        #print("#######################################################################")
        #print("###############           USER INPUT                 ##################")
        #print("#######################################################################")
    else:
        applied_force = fuzzy_response

    #
    # Wyświetl stan środowiska oraz wartość odpowiedzi regulatora na ten stan.
    #print(
    #    f"cpos={cart_position:8.4f}, cvel={cart_velocity:8.4f}, pang={pole_angle:8.4f}, tvel={tip_velocity:8.4f}, force={applied_force:8.4f}")

    #
    # Wykonaj krok symulacji
    env.step(applied_force)

    #
    # Pokaż kotku co masz w środku
    env.render()

#
# Zostaw ten patyk!
env.close()

