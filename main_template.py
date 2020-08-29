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


#
# force definitions
#

slight_push = 3
strong_push = 7



#
# pole angle 
#
pole_angle_range = np.arange(-6, 6, 0.01)

pole_angle_target = 0;

pole_angle_l2 = pole_angle_target - 0.2
pole_angle_l1 = pole_angle_target - 0.1
pole_angle_r1 = pole_angle_target + 0.1
pole_angle_r2 = pole_angle_target + 0.2

pole_angle_vertical         = fuzz.trimf(pole_angle_range, [pole_angle_l1, pole_angle_target, pole_angle_r1])

pole_angle_slight_left      = fuzz.trimf(pole_angle_range, [pole_angle_l2, pole_angle_l1, pole_angle_target])
pole_angle_left             = fuzz.trapmf(pole_angle_range, [-10, -10, pole_angle_l2, pole_angle_l1])

pole_angle_slight_right     = fuzz.trimf(pole_angle_range, [pole_angle_target, pole_angle_r1, pole_angle_r2])
pole_angle_right            = fuzz.trapmf(pole_angle_range, [pole_angle_r1, pole_angle_r2, 10, 10])



#
# cart_position
#
cart_position_target = 1

cart_position_range = np.arange(-10, 10, 0.01)

cart_position_l2 = cart_position_target - 0.25
cart_position_l1 = cart_position_target - 0.10
cart_position_r1 = cart_position_target + 0.10
cart_position_r2 = cart_position_target + 0.25

cart_position_desired         = fuzz.trimf(cart_position_range, [cart_position_l1, cart_position_target, cart_position_r1])

cart_position_slight_left      = fuzz.trimf(cart_position_range, [cart_position_l2, cart_position_l1, cart_position_target])
cart_position_left             = fuzz.trapmf(cart_position_range, [-10, -10, cart_position_l2, cart_position_l1])

cart_position_slight_right     = fuzz.trimf(cart_position_range, [cart_position_target, cart_position_r1, cart_position_r2])
cart_position_right            = fuzz.trapmf(cart_position_range, [cart_position_r1, cart_position_r2, 10, 10])


#
# cart_velocity
#
cart_velocity_target = 0

cart_velocity_range = np.arange(-10, 10, 0.01)

cart_velocity_l2 = cart_velocity_target - 1
cart_velocity_l1 = cart_velocity_target - 0.5
cart_velocity_r1 = cart_velocity_target + 0.5
cart_velocity_r2 = cart_velocity_target + 1

cart_velocity_zero          = fuzz.trimf(cart_velocity_range, [cart_velocity_l1, cart_velocity_target, cart_velocity_r1])

cart_velocity_slight_left   = fuzz.trimf(cart_velocity_range, [cart_velocity_l2, cart_velocity_l1, cart_velocity_target])
cart_velocity_left          = fuzz.trapmf(cart_velocity_range, [-10, -10, cart_velocity_l2, cart_velocity_l1])

cart_velocity_slight_right  = fuzz.trimf(cart_velocity_range, [cart_velocity_target, cart_velocity_r1, cart_velocity_r2])
cart_velocity_right         = fuzz.trapmf(cart_velocity_range, [cart_velocity_r1, cart_velocity_r2, 10, 10])



"""

1. Określ dziedzinę dla każdej zmiennej lingwistycznej. Każda zmienna ma własną dziedzinę.
2. Zdefiniuj funkcje przynależności dla wybranych przez siebie zmiennych lingwistycznych.
3. Wyświetl je, w celach diagnostycznych.

Przykład wyświetlania:
"""
if True:
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 9))

    ax0.plot(pole_angle_range, pole_angle_left, 'b', linewidth=1.5, label='Left')
    ax0.plot(pole_angle_range, pole_angle_slight_left, 'c', linewidth=1.5, label='slight left')

    ax0.plot(pole_angle_range, pole_angle_vertical, 'k', linewidth=1.5, label='Zero')

    ax0.plot(pole_angle_range, pole_angle_slight_right, 'm', linewidth=1.5, label='slight right')
    ax0.plot(pole_angle_range, pole_angle_right, 'r', linewidth=1.5, label='Right')
    ax0.set_title('Pole angle')
    ax0.legend()


    ax1.plot(cart_position_range, cart_position_left, 'b', linewidth=1.5, label='Left')
    ax1.plot(cart_position_range, cart_position_slight_left, 'c', linewidth=1.5, label='slight left')

    ax1.plot(cart_position_range, cart_position_desired, 'k', linewidth=1.5, label='Zero')

    ax1.plot(cart_position_range, cart_position_slight_right, 'm', linewidth=1.5, label='slight right')
    ax1.plot(cart_position_range, cart_position_right, 'r', linewidth=1.5, label='Right')
    ax1.set_title('Cart position')
    ax1.legend()





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

    # pole angle
    is_pole_angle_left =            fuzz.interp_membership(pole_angle_range, pole_angle_left,           pole_angle)
    is_pole_angle_slight_left =     fuzz.interp_membership(pole_angle_range, pole_angle_slight_left,    pole_angle)
    is_pole_angle_vertical =        fuzz.interp_membership(pole_angle_range, pole_angle_vertical,       pole_angle)
    is_pole_angle_right =           fuzz.interp_membership(pole_angle_range, pole_angle_right,          pole_angle)
    is_pole_angle_slight_right =    fuzz.interp_membership(pole_angle_range, pole_angle_slight_right,   pole_angle)

    # cart position
    is_cart_position_left =         fuzz.interp_membership(cart_position_range, cart_position_left,         cart_position)
    is_cart_position_slight_left =  fuzz.interp_membership(cart_position_range, cart_position_slight_left,  cart_position)
    is_cart_position_desired =      fuzz.interp_membership(cart_position_range, cart_position_desired,      cart_position)
    is_cart_position_slight_right = fuzz.interp_membership(cart_position_range, cart_position_slight_right, cart_position)
    is_cart_position_right =        fuzz.interp_membership(cart_position_range, cart_position_right,        cart_position)

    # cart velocity
    is_cart_velocity_left =         fuzz.interp_membership(cart_velocity_range, cart_velocity_left,         cart_velocity)
    is_cart_velocity_slight_left =  fuzz.interp_membership(cart_velocity_range, cart_velocity_slight_left,  cart_velocity)
    is_cart_velocity_zero =         fuzz.interp_membership(cart_velocity_range, cart_velocity_zero,      cart_velocity)
    is_cart_velocity_slight_right = fuzz.interp_membership(cart_velocity_range, cart_velocity_slight_right, cart_velocity)
    is_cart_velocity_right =        fuzz.interp_membership(cart_velocity_range, cart_velocity_right,        cart_velocity)


    # init actions
    actions = Actions()

    # rules for pole angle

    """
    #if pole is vertical then apply no force
    actions.actions['idle'] *= is_pole_angle_vertical*CartForce.IDLE_FORCE

    #if pole is slightly left then apply force slightly left
    actions.actions['slight_left'] *= is_pole_angle_slight_left

    #if pole is left then apply force left
    actions.actions['left'] *= is_pole_angle_left


    #if pole is slightly right then apply force slightly right
    actions.actions['slight_right'] *= is_pole_angle_slight_right

    #if pole is right then apply force right
    actions.actions['right'] *= is_pole_angle_right


    
    
    """
    # rules for cart position and cart velocity
    
    #if cart is in desired poition AND cart velocity is zero
    #   then apply no force
    #actions.actions['idle'] *= np.fmin(is_cart_position_desired, is_cart_velocity_zero)

    #if cart is in desired poition AND cart velocity is right 
    # then apply left force
    #if 
    #   (cart is slightly right of desired position OR cart is right of desired position) 
    #   AND 
    #   (cart velocity is zero OR cart velocity is right OR cart velocity is slightly right)
    #   then apply left force
    left1 = np.fmin(np.fmax(is_cart_position_right, is_cart_position_slight_right), np.fmax(is_cart_velocity_right, is_cart_velocity_slight_right))
    left2 = np.fmin(is_cart_position_desired, is_cart_velocity_right)
    actions.actions['left'] *= np.fmax(left1, left2)


    #if cart is in desired poition AND cart velocity is left 
    # then apply right force
    #if 
    #   (cart is slightly left of desired position OR cart is left of desired position) 
    #   AND 
    #   (cart velocity is zero OR cart velocity is left OR cart velocity is slightly left)
    #   then apply right force
    right1 = np.fmin(np.fmax(is_cart_position_left, is_cart_position_slight_left), np.fmax(is_cart_velocity_left, is_cart_velocity_slight_left))
    right2 = np.fmin(is_cart_position_desired, is_cart_velocity_left)
    actions.actions['right'] *= np.fmax(right1, right2) 

    #if cart is in desired poition AND cart velocity is slight right 
    # then apply slight left force
    actions.actions['slight_left'] *= np.fmin(is_cart_position_desired, is_cart_velocity_slight_right)
    #if cart is in desired poition AND cart velocity is slight left 
    # then apply slight right force
    actions.actions['slight_right'] *= np.fmin(is_cart_position_desired, is_cart_velocity_slight_left)





    #if 
    #   (cart is slightly left of desired position OR cart is left of desired position) 
    #   AND 
    #   (cart velocity is slightly right OR cart velocity is right)
    #   then apply no force
    #actions.actions['idle'] *=  np.fmin(np.fmax(is_cart_position_slight_left, is_cart_position_left), np.fmax(is_cart_velocity_right, is_cart_velocity_slight_right))
    


    #if 
    #   (cart is slightly right of desired position OR cart is right of desired position) 
    #   AND 
    #   (cart velocity is slightly left OR cart velocity is right)
    #   then apply no force
    #actions.actions['idle'] *=  np.fmin(np.fmax(is_cart_position_slight_right, is_cart_position_right), np.fmax(is_cart_velocity_left, is_cart_velocity_slight_left))
    


    





    

    
    
    
    fuzzy_response = actions.getAction()
    #
    # KONIEC algorytmu regulacji
    #########################

    # Jeżeli użytkownik chce przesunąć wózek, to jego polecenie ma wyższy priorytet
    if control.UserForce is not None:
        applied_force = control.UserForce
        control.UserForce = None
    else:
        applied_force = fuzzy_response

    #
    # Wyświetl stan środowiska oraz wartość odpowiedzi regulatora na ten stan.
    print(
        f"cpos={cart_position:8.4f}, cvel={cart_velocity:8.4f}, pang={pole_angle:8.4f}, tvel={tip_velocity:8.4f}, force={applied_force:8.4f}")

    #
    # Wykonaj krok symulacji
    env.step(applied_force)

    #
    # Pokaż kotku co masz w środku
    env.render()

#
# Zostaw ten patyk!
env.close()

