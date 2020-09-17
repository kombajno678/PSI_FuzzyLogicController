#
# Podstawy Sztucznej Inteligencji, IIS 2020
# Autor: Tomasz Jaworski
# Opis: Szablon kodu do stabilizacji odwróconego wahadła (patyka) w pozycji pionowej podczas ruchu wózka.
#

import gym # Instalacja: https://github.com/openai/gym
import time
from helper import HumanControl, Keys, CartForce, Response
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
# helper/alias functions
#

def OR(a, b):
    return np.fmax(a, b)

def AND(a, b):
    return np.fmin(a, b)

"""
1. Określ dziedzinę dla każdej zmiennej lingwistycznej. Każda zmienna ma własną dziedzinę.
2. Zdefiniuj funkcje przynależności dla wybranych przez siebie zmiennych lingwistycznych.
"""
#
# pole angle 
#
pole_angle_target = 0;
pole_angle_d = 0.12

pole_angle_range_max = 3
pole_angle_range = np.arange(-pole_angle_range_max, pole_angle_range_max, 0.01)

pole_angle_l1 = pole_angle_target - pole_angle_d
pole_angle_r1 = pole_angle_target + pole_angle_d

pole_angle_left     = fuzz.trapmf(pole_angle_range, [-pole_angle_range_max, -pole_angle_range_max, pole_angle_l1,        pole_angle_target])
pole_angle_vertical = fuzz.trimf(pole_angle_range,  [ pole_angle_l1,         pole_angle_target,    pole_angle_r1])
pole_angle_right    = fuzz.trapmf(pole_angle_range, [ pole_angle_target,     pole_angle_r1,        pole_angle_range_max, pole_angle_range_max])
#
# cart_position
#
cart_position_target = 0

cart_position_target = float(input("input cart's target position (0 => middle, -1 => left , +1 => right)"))
#cart_position_target = 0.0

cart_position_d = 0.40

cart_position_range_max = 5
cart_position_range = np.arange(-cart_position_range_max, cart_position_range_max, 0.01)

cart_position_l1 = cart_position_target - cart_position_d
cart_position_r1 = cart_position_target + cart_position_d

cart_position_left    = fuzz.trapmf(cart_position_range, [-cart_position_range_max, -cart_position_range_max, cart_position_l1,        cart_position_target])
cart_position_desired = fuzz.trimf(cart_position_range,  [ cart_position_l1,         cart_position_target,    cart_position_r1])
cart_position_right   = fuzz.trapmf(cart_position_range, [ cart_position_target,     cart_position_r1,        cart_position_range_max, cart_position_range_max])


#
# cart_velocity
#
cart_velocity_target = 0
cart_velocity_d = 0.8

cart_velocity_range_max = 10
cart_velocity_range = np.arange(-cart_velocity_range_max, cart_velocity_range_max, 0.01)

cart_velocity_l1 = cart_velocity_target - cart_velocity_d
cart_velocity_r1 = cart_velocity_target + cart_velocity_d

cart_velocity_left  = fuzz.trapmf(cart_velocity_range, [-cart_velocity_range_max, -cart_velocity_range_max, cart_velocity_l1,        cart_velocity_target])
cart_velocity_zero  = fuzz.trimf(cart_velocity_range,  [ cart_velocity_l1,         cart_velocity_target,    cart_velocity_r1])
cart_velocity_right = fuzz.trapmf(cart_velocity_range, [ cart_velocity_target,     cart_velocity_r1,        cart_velocity_range_max, cart_velocity_range_max])

#
# defuzze parameters
#

response_force = 50
response_method = 'centroid'


"""
3. Wyświetl je, w celach diagnostycznych.
"""
if True:
    temp_response = Response(response_force, response_method)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

    ax0.plot(pole_angle_range, pole_angle_left, 'b', linewidth=1.5, label='Left')
    ax0.plot(pole_angle_range, pole_angle_vertical, 'k', linewidth=1.5, label='Vertical')
    ax0.plot(pole_angle_range, pole_angle_right, 'r', linewidth=1.5, label='Right')
    ax0.set_title('Pole angle')
    ax0.legend()

    ax1.plot(cart_position_range, cart_position_left, 'b', linewidth=1.5, label='Left')
    ax1.plot(cart_position_range, cart_position_desired, 'k', linewidth=1.5, label='Target')
    ax1.plot(cart_position_range, cart_position_right, 'r', linewidth=1.5, label='Right')
    ax1.set_title('Cart position')
    ax1.legend()
    
    ax2.plot(cart_velocity_range, cart_velocity_left, 'b', linewidth=1.5, label='Left')
    ax2.plot(cart_velocity_range, cart_velocity_zero, 'k', linewidth=1.5, label='Zero')
    ax2.plot(cart_velocity_range, cart_velocity_right, 'r', linewidth=1.5, label='Right')
    ax2.set_title('Cart velocity')
    ax2.legend()

    ax3.plot(temp_response.force_range, temp_response.force_left , 'b', linewidth=1.5, label='Left')
    ax3.plot(temp_response.force_range, temp_response.force_idle , 'k', linewidth=1.5, label='Idle')
    ax3.plot(temp_response.force_range, temp_response.force_right, 'r', linewidth=1.5, label='Right')
    ax3.set_title('Response force')
    ax3.legend()

    plt.tight_layout()
    plt.show()
#########################################################
# KONIEC KODU INICJUJĄCEGO
#########################################################


#
# Główna pętla symulacji
#
i = 0
avg_cart_pos = 0
min_cart_pos = 5
max_cart_pos = -5
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
        i = 0
        avg_cart_pos = 0
        min_cart_pos = 5
        max_cart_pos = -5


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

    #
    # reset if cart out of screen or pole down
    #
    if pole_angle > 3 or pole_angle < -3 or cart_position > 5 or cart_position < -5:
        env.reset()
        i = 0
        avg_cart_pos = 0
        min_cart_pos = 5
        max_cart_pos = -5
        continue


    avg_cart_pos = (avg_cart_pos*i + cart_position) / (i+1)
    i += 1

    if cart_position > max_cart_pos:
        max_cart_pos = cart_position
    
    if cart_position < min_cart_pos:
        min_cart_pos = cart_position

    fuzzy_response = CartForce.IDLE_FORCE # do zmiennej fuzzy_response zapisz wartość siły, jaką chcesz przyłożyć do wózka.

    """
    1. Przeprowadź etap rozmywania, w którym dla wartości zmierzonych wyznaczone zostaną ich przynależności do poszczególnych
       zmiennych lingwistycznych. Jedno fizyczne wejście (źródło wartości zmierzonych, np. położenie wózka) posiada własną
       zmienną lingwistyczną.
       
       Sprawdź funkcję interp_membership
    """
    
    # pole angle
    is_pole_angle_left     = fuzz.interp_membership(pole_angle_range, pole_angle_left,     pole_angle)
    is_pole_angle_vertical = fuzz.interp_membership(pole_angle_range, pole_angle_vertical, pole_angle)
    is_pole_angle_right    = fuzz.interp_membership(pole_angle_range, pole_angle_right,    pole_angle)
    #print(
    #    f"pole angle [{is_pole_angle_left:8.4f} {is_pole_angle_vertical:8.4f} {is_pole_angle_right:8.4f}]")

    # cart position
    is_cart_position_left    = fuzz.interp_membership(cart_position_range, cart_position_left,    cart_position)
    is_cart_position_desired = fuzz.interp_membership(cart_position_range, cart_position_desired, cart_position)
    is_cart_position_right   = fuzz.interp_membership(cart_position_range, cart_position_right,   cart_position)
    #print(
    #    f"cart posit [{is_cart_position_left:8.4f} {is_cart_position_desired:8.4f} {is_cart_position_right:8.4f}]")

    # cart velocity
    is_cart_velocity_left  = fuzz.interp_membership(cart_velocity_range, cart_velocity_left,  cart_velocity)
    is_cart_velocity_zero  = fuzz.interp_membership(cart_velocity_range, cart_velocity_zero,  cart_velocity)
    is_cart_velocity_right = fuzz.interp_membership(cart_velocity_range, cart_velocity_right, cart_velocity)
    #print(
    #    f"cart veloc [{is_cart_velocity_left:8.4f} {is_cart_velocity_zero:8.4f} {is_cart_velocity_right:8.4f}]")


    # init actions
    r = Response(response_force, response_method)


    """
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
    """

    left1 = AND(is_pole_angle_vertical, is_cart_position_left)
    left2 = is_pole_angle_left
    left3 = AND(is_pole_angle_left, AND(is_cart_velocity_right, OR(is_cart_position_left, is_cart_position_desired)))

    r.actions['left'] = OR(left3, OR(left1, left2))


    idle1 = is_pole_angle_vertical
    idle2 = AND(is_cart_position_desired, is_cart_velocity_zero)
    idle3 = AND(AND(is_pole_angle_vertical, is_cart_position_left), is_cart_velocity_right)
    idle4 = AND(AND(is_pole_angle_vertical, is_cart_position_right), is_cart_velocity_left)
    
    r.actions['idle'] = OR(OR(idle1, idle2), OR(idle3, idle4))



    right1 = AND(is_pole_angle_vertical, is_cart_position_right)
    right2 = is_pole_angle_right
    right3 = AND(is_pole_angle_right, AND(is_cart_velocity_left, OR(is_cart_position_right, is_cart_position_desired)))

    r.actions['right'] = OR(right3, OR(right1, right2))

   

    #print(f"{left1:1.2f} {left2:1.2f} ||| {idle1:1.2f} {idle2:1.2f} {idle3:1.2f} {idle4:1.2f} ||| {right1:1.2f} {right2:1.2f}")
    
    #print(r.actions)
    fuzzy_response = r.defuzze()
    #print(f"fuzzy_response = {fuzzy_response}")
    cart_pod_diff = cart_position_target - avg_cart_pos


    if i % 10 == 0 or True:
        print(f"{i:8}  >CART_POS:[AVG = {avg_cart_pos:6.3f}; TARGET = {cart_position_target}; MIN = {min_cart_pos:6.3f}; MAX = {max_cart_pos:6.3f}] RES = {fuzzy_response:6.2f} ||| {left1:1.2f} {left2:1.2f} {left3:1.2f} | {idle1:1.2f} {idle2:1.2f} {idle3:1.2f} {idle4:1.2f} | {right1:1.2f} {right2:1.2f} {right3:1.2f}")
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

