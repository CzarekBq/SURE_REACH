import numpy as np
import pygame
import sys
import pickle
import random
np.set_printoptions(threshold=sys.maxsize)
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation

global g

l_shoulder = 140
l_elbow = 70
l_wrist = 42

    #proposed lengths l_shoulder = 140, l_elbow = 70, l_wrist = 42


    # initial position of the sliders
sx1 = 545
sx2 = sx1
sx3 = sx1

    # display dimensions
WIDTH_ARM = 504
WIDTH_CTRL = 300
HEIGHT_ARM = 504
WIDTH = WIDTH_ARM + WIDTH_CTRL
HEIGHT = HEIGHT_ARM

    #arm movement
g = np.pi/12 # gain factor of about 15 degrees







def restart_weights(W_PM_erase = False, W_is_erase = False, file_name='', folder=''):
    
    if W_PM_erase:
        W_PM = np.zeros((405,441))
        np.savetxt(folder+'W_PM'+file_name+'.csv', W_PM, delimiter=',')
        
        print('W_PM succesfully restarted!')
        
    if W_is_erase:
        for i in range(6):
            W_i = np.zeros((405,405))
            r = np.zeros((405,1))
            
            name_W = folder+'W_' + str(i) + file_name +'.csv'
            name_r = folder+'r_' + str(i) + file_name +'.csv'
            
            np.savetxt(name_W, W_i, delimiter=',')
            np.savetxt(name_r, r, delimiter=',')
        
        print('W_is and rs are succesfully restarted!')
            
            


def hi(hix,hiy,x,y): #correct
    '''
    input:
    hix, hiy (scalars)- given coordinates of a hand space neuron
    x, y (scalar)- given coordinates of the tip of the arm 
    
    output:
    the activation around that hand space neuron
    '''
    scale = 0.1*(l_shoulder + l_elbow + l_wrist)
    
    disX = 1-abs(x-hix)/scale
    disY = 1-abs(y-hiy)/scale
    hi = max(0,disX)*max(0,disY)
    return hi


def generate_h_vector(tip, neurons = 21):
    '''
    input:
    [x, y] numpy array- tuple of the xy coordinates of the tip of the arm
    
    output:
    h (neruons*neurons, 1) - the handspace vector
    '''
    
    workspace_dim = WIDTH_ARM
    
    [x, y] = tip
    cell = int(workspace_dim / neurons)
    
    h = np.zeros((neurons*neurons,1))
    
    for a in range(neurons):
        for b in range(neurons):
            h[a*neurons+b] = hi(a*cell, b*cell, x, y)
    
    return h


def h_to_space(h_arg):
    '''
    description: changes the index of the one hot handspace vector into corrdinates in space
    
    input:
    h_arg (scalar) - the index of the value 1.0 in the vector h
    
    output:
    (x,y) - tuple of the space coordinates where the handspace vector is activated
    '''
    y = (h_arg%21)*24
    x = ((h_arg - (h_arg%21))//21)*24
    return np.array([x,y])

def pi(phi_arr, p_arr):
    '''
    input:
    phi_arr (3,1) - three selected angles of the posture space (np.array)
    p_arr (3,1) - three joint angles of the arm (np.array)
    
    output:
    acc (scalar) - posture space neuron activation
    
    *all units are in radians
    '''
    
    dp = phi_arr - p_arr
    ap = (1.0 - abs(dp)/(np.pi/4))
    acc = 1
    for p in ap.flat:
        acc = acc * max(p, 0.0)
    return acc

def generate_p_vector(sa, ea, wa):
    '''
    description: the function generates the whole posture space vector. the way a 3d joint space gets mapped onto a vector is
    by jumping by 5 more entries after 5 wrist position and similarily jumping over 5*9=45 vector position after all elbow-wrist
    combination at a given shoulder angle in posture neuron.
    
    input:
    the current joint angles of the arm: sa- shoulder angle, ea- elbow angle, wa- wrist angle 
    
    output:
    p (405, 1) - posture space vector
    '''
    
    p_arr = np.array([[sa],[ea],[wa]])
    
    p = np.zeros((9*9*5, 1))
    
    for i in range(-4,5):
        for j in range(-4,5):
            for k in range(5):
                
                phi_arr = np.array([[np.pi/4 *i],[np.pi/4 *j],[np.pi/4 *k]])
                #print("index: "+str(45*(i+4)+5*(j+4)+k)+"   phi_arr:"+ str(phi_arr)+"   actvation: "+str(pi(phi_arr, p_arr)))
                p[45*(i+4)+5*(j+4)+k,0] = pi(phi_arr, p_arr)
    
    return p

def p_to_angles(p_arg):
    '''
    description: turns the one hot posture space vector into the actual angles of the joints of the arm
    
    input: 0 45? 0-44 has same elbow angle
    p_arg (scalar) - the max arg of some one hot posture space vector
    '''
    wrist = (p_arg%5)*np.pi/4
    elbow = ((p_arg - p_arg//45*45)//5 - 4)*np.pi/4
    shoulder = ((p_arg//45)-4)*np.pi/4
    return np.array([shoulder, elbow, wrist])


def update_W_PM(w_pm, p, h, epsilon = 0.001):
    '''
    description: the function updates the weights of the posture memory matrix W_PM using hebbian learning principle
    
    input:
    w_pm (p.size, p.size) - posture memory matrix before hebbian learning weight adjustment
    h (441, 1) - hand space vector
    p (405, 1) - posutre space vector
    
    output:
    updated_w_pm - the same matrix with its weights adjusted
    '''
    
    #changed from np.multiply to np.matmul !!!
    updated_w_pm = w_pm + epsilon * np.matmul(p, h.T)
    
    return updated_w_pm


def arm_forward_kinematics(shoulder, sa, ea, wa):
    '''
    description: moves the arm given the position of the base and the angles of the joints using forward kinematics.
    link lengths are given at the start of the notebook as l_shoulder, l_elbow and l_wrist
    
    input:
    shoulder - [x,y] coordinate tuple of the base of the arm
    sa - angle of the shoulder joint
    ea - angle of the elbow joint
    wa - angle of the wrist joint
    
    output:
    (shoulder, elbow, wrist, tip) - tuple of numpy arrays of coorinates of each joint of the arm
    '''
    [sx, sy] = shoulder
    elbow = np.array([int(sx + np.cos(sa)*l_shoulder), int(sy + np.sin(sa)*l_shoulder)])
    [ex, ey] = elbow
    wrist = np.array([int(np.cos(ea+sa)*l_elbow + sx + np.cos(sa)*l_shoulder), int(np.sin(ea+sa)*l_elbow + sy + np.sin(sa)*l_shoulder)])
    [wx, wy] = wrist
    tip = np.array([int(np.cos(wa+ea+sa)*l_wrist + np.cos(ea+sa)*l_elbow + sx + np.cos(sa)*l_shoulder) ,int(np.sin(wa+ea+sa)*l_wrist + np.sin(ea+sa)*l_elbow + sy + np.sin(sa)*l_shoulder )])
    return [shoulder, elbow, wrist, tip]



'''
#function that allows for both commands to fire
def random_motor_command():
    
    ys = []
    
    for i in range(6):
        if random.random() <= 0.3:
            y = 1.0
        else:
            y = 0.0
        
        ys.append(y)
    
    return ys'''

def random_motor_command():
    
    ys = []
    
    for i in range(3):
        if random.random() <= 0.3:
            y = 1.0
        else:
            y = 0.0
        if random.random() <= 0.5:
            ys.append(y)
            ys.append(0.0)
        else:
            ys.append(0.0)
            ys.append(y)
    
    return ys

def gate(sa,ea,wa):
    
    passs = False
    
    if  -np.pi > sa:
        passs = True
    elif np.pi < sa:
        passs = True

    if -np.pi > ea:
        passs = True
    elif np.pi < ea:
        passs = True

    if  0 > wa:
        passs = True
    elif np.pi < wa:
        passs = True
    
    return passs


def motor_babbling(sa, ea, wa, constraints = True):
    
    go = True

    while go and constraints:

        ys = random_motor_command()

        while sum(ys) == 0.0:
            ys = random_motor_command()

        angs = [sa, ea, wa]
        new_angs = []

        for a in range(3):
            new_ang = angs[a] + g * (ys[2*a] - ys[2*a + 1]) #[ postive/ ccw, negative/ cw, --, --]
            new_angs.append(new_ang)

        [new_sa, new_ea, new_wa] = new_angs

        go = gate(new_sa, new_ea, new_wa)
    
    
    return ys, new_sa, new_ea, new_wa



def leaky_integrator_r(y, p, r, rho=0.1):
    '''
description: the function calculates the new leaky integrator vector given the past posture and the motor command that lead to it
input:
y - a single i motor command
p (405, 1) -
r (405, 1) - the previous leaky integrator vector
output:
new_r (405, 1) - the updated posture trace 
    '''
    new_r = y * p + rho * r
   
    return new_r

def update_delta(delta, i, iters = 10**6):
    '''
    WRONG DECAY! IT'S LINEAR NOT EXPONENTIAL
    a = - 9 / (10**8 - 10 **2)
    new_delta = a*delta + 0.1
    '''
    #new_delta = delta * (10**(-10**(-6)))**i
    new_delta = delta * (10**(-(iters)**(-1)))**i

    
    return new_delta

def update_W_i(W_i, r, p, i, iters, delta=0.1, theta=0.1):
    '''
description: the function calculates the new synaptic weights of the movement trace matrix.
Input:
W_i (405, 405) - the previous movement trace matrix
r (405, 1) - the previous movement trace vector
p (405,1) - the previous posture
delta, theta - coefficients
    '''
#changed from np.multiply to np.matmul !!!
    new_W_i = W_i + update_delta(delta, i, iters = iters) * np.matmul(r,p.T) * (theta-W_i)
    
    return new_W_i

def load_matrices(file_name = '', folder = ''):
    '''
    description: loads matrix weights of W_PM, W_i's and r_i's from .csv files
    '''
    
    W_PM = np.loadtxt(folder+'W_PM'+file_name+'.csv', delimiter = ',')

    W_0 = np.loadtxt(folder+'W_0'+file_name+'.csv', delimiter = ',')
    W_1 = np.loadtxt(folder+'W_1'+file_name+'.csv', delimiter = ',')
    W_2 = np.loadtxt(folder+'W_2'+file_name+'.csv', delimiter = ',')
    W_3 = np.loadtxt(folder+'W_3'+file_name+'.csv', delimiter = ',')
    W_4 = np.loadtxt(folder+'W_4'+file_name+'.csv', delimiter = ',')
    W_5 = np.loadtxt(folder+'W_5'+file_name+'.csv', delimiter = ',')

    W_is = [W_0, W_1, W_2, W_3, W_4, W_5]

    r_0 = np.loadtxt(folder+'r_0'+file_name+'.csv', delimiter = ',')
    r_1 = np.loadtxt(folder+'r_1'+file_name+'.csv', delimiter = ',')
    r_2 = np.loadtxt(folder+'r_2'+file_name+'.csv', delimiter = ',')
    r_3 = np.loadtxt(folder+'r_3'+file_name+'.csv', delimiter = ',')
    r_4 = np.loadtxt(folder+'r_4'+file_name+'.csv', delimiter = ',')
    r_5 = np.loadtxt(folder+'r_5'+file_name+'.csv', delimiter = ',')

    r_0 = np.reshape(r_0, (405,1))
    r_1 = np.reshape(r_1, (405,1))
    r_2 = np.reshape(r_2, (405,1))
    r_3= np.reshape(r_3, (405,1))
    r_4 = np.reshape(r_4, (405,1))
    r_5 = np.reshape(r_5, (405,1))

    rs = [r_0, r_1, r_2, r_3, r_4, r_5]
    
    return W_PM, W_is, rs

def save_matrices(W_PM, W_is, rs, file_name = '', folder = ''):
    '''
    description: saves (numpy) matrix weights of W_PM, W_i's and r_i's into .csv files
    '''
    
    np.savetxt(folder+'W_PM'+file_name+'.csv', W_PM, delimiter=',')

    np.savetxt(folder+'W_0'+file_name+'.csv', W_is[0], delimiter = ',')
    np.savetxt(folder+'W_1'+file_name+'.csv', W_is[1], delimiter = ',')
    np.savetxt(folder+'W_2'+file_name+'.csv', W_is[2], delimiter = ',')
    np.savetxt(folder+'W_3'+file_name+'.csv', W_is[3], delimiter = ',')
    np.savetxt(folder+'W_4'+file_name+'.csv', W_is[4], delimiter = ',')
    np.savetxt(folder+'W_5'+file_name+'.csv', W_is[5], delimiter = ',')

    np.savetxt(folder+'r_0'+file_name+'.csv', rs[0], delimiter = ',')
    np.savetxt(folder+'r_1'+file_name+'.csv', rs[1], delimiter = ',')
    np.savetxt(folder+'r_2'+file_name+'.csv', rs[2], delimiter = ',')
    np.savetxt(folder+'r_3'+file_name+'.csv', rs[3], delimiter = ',')
    np.savetxt(folder+'r_4'+file_name+'.csv', rs[4], delimiter = ',')
    np.savetxt(folder+'r_5'+file_name+'.csv', rs[5], delimiter = ',')

def normalize_vec(vector):
    '''
    description: the function normalizes a given vector so that its all entries add up to 1.0
    
    input:
    vector - a columnar numpy vector
    
    output:
    new_vec - normalized columnar numpy vector
    '''
    
    vec_sum = np.sum(vector)
    
    if vec_sum != 0:
        new_vec = vector / vec_sum
    else:
        new_vec = vector
    
    return new_vec


def update_as(a, p_goal, W_is, beta= 0.172, gamma= 0.434, v = [1, 1, 1, 1, 1, 1]):
    '''
    description: updates the weights of a columnar vector so sensory-to-motor mappings.
    
    Input:
    a [(405,1)x6] - list of previous activation maps (list of numpy arrays)
    p_goal (405,1) - desired goal posture to be acheived (numpy array)
    W_is [(405, 405)x6] - list of previous motor and posture trace matrices (list of numpy arrays)
    
    output:
    new_as [(405,1)x6] - 
    '''
    
    new_as = []
    
    for i in range(6):
        ap = np.maximum(beta * (gamma * (sum(a) - a[i])/5 + (1-gamma) * a[i]), p_goal)
        
        #print("the averaging step:\n"+str(ap))

        #changed from np.multiply to np.matmul !!!
        app = ap + np.matmul(W_is[i], ap)
        
        a_norm = v[i] *  normalize_vec(app)
        
        new_as.append(a_norm)
        
    return new_as

def update_as_obs(a, p_goal, p_deac, W_is, beta= 0.172, gamma= 0.434, v = [1, 1, 1, 1, 1, 1]):
    '''
    description: updates the weights of a columnar vector so sensory-to-motor mappings.
    
    Input:
    a [(405,1)x6] - list of previous activation maps (list of numpy arrays)
    p_goal (405,1) - desired goal posture to be acheived (numpy array)
    p_deac (405,1) - obstacle representation in posture space
    W_is [(405, 405)x6] - list of previous motor and posture trace matrices (list of numpy arrays)
    
    output:
    new_as [(405,1)x6] - 
    '''
    
    new_as = []
    
    for i in range(6):
        ap = np.maximum(beta * (gamma * (sum(a) - a[i])/5 + (1-gamma) * a[i]), p_goal)
        
        #print("the averaging step:\n"+str(ap))
        for j in range(405):
            if p_deac[j,0] > 0.0 and ap[j,0] > 0.01:
                ap[j,0] = 0.0

        app = ap + np.matmul(W_is[i], ap)
        
        for j in range(405):
            if p_deac[j,0] > 0.0 and app[j,0] > 0.01:
                app[j,0] = 0.0

        a_norm = v[i] *  normalize_vec(app)
        
        new_as.append(a_norm)
        
    return new_as
# function for testing
def update_as_avg(a, p_goal, diagnostics = False, beta= 0.172, gamma= 0.434): #dev tool for investigating the averaging part

    a_avg = []
    
    for i in range(6):
        ap = np.maximum(beta * (gamma * (sum(a) - a[i])/5 + (1-gamma) * a[i]), p_goal)
        if diagnostics: 
            #print("The averaging of the mapping:"+ str(sum(a) - a[i])/5)
            print("sum of avgs: " + str(sum(sum(a) - a[i])))
            #print("The copied mapping: "+ str(a[i]))
            print("sum of a: " + str(sum(a[i])))
        a_avg.append(ap)
    
    return a_avg

# function for testing
def update_as_mix(a_avg, W_is, diagnostics = False): #takes a list of a's that were only averaged but not mixed in
    new_as = []
    i = 0
    for ap in a_avg:
        app = ap + np.matmul(W_is[i], ap)
        if diagnostics:
            print("sum of the mixed vector mapping: " + str(sum(app)))
        i = i + 1
        a_norm = normalize_vec(app)
        if diagnostics:
            print("sum pf the normalized and mixed vectors: "+str(sum(a_norm)))
        new_as.append(a_norm)
    
    return new_as


def goal_motor_commands(a, p):
    '''
    description: given the activation mapping vector and the current posture the function generates appropriate
    motor commands to get the arm closer to the goal posture.
    
    input:
    a [(405,1)x6] - list of previous activation maps (list of numpy arrays)
    p (405,1) - the current posture space neuron of the arm
    
    output:
    y_finals [scalar x6]- a list of motors commands (scalars)
    '''
    
    yS = []
    
    for i in range(6):
        y = np.matmul(p.T, a[i])
        yS.append(y.flat[0])
      
    y_nets = []
    
    y_squared_sum = sum(yS[k]**2 for k in range(len(yS)))
    
    
    for j in range(6):
        if y_squared_sum != 0:
            y_net = yS[j]**2 / y_squared_sum
        else:
            y_net = yS[j]
        y_nets.append(y_net.flat[0])
    
    #print('y_net: '+ str(y_nets))
    
    y_nets_star = y_nets.copy()
    
    for k in range(3):
        if (y_nets[2*k] - y_nets[2*k + 1]) > 0:
            y_nets_star[2*k] = y_nets[2*k] - y_nets[2*k + 1]
            y_nets_star[2*k + 1] = 0
        else:
            y_nets_star[2*k] = 0
            y_nets_star[2*k + 1] = y_nets[2*k + 1] - y_nets[2*k]
                    
    #print('y_nets_star: '+ str(y_nets_star))
    
    y_nets_star_sum = sum(y_nets_star)
    #print(y_nets_star_sum)
    
    y_finals = []
        
    #for x in range(6):
    #    if y_nets_star_sum != 0.0:
    #        y_final = g * y_nets_star[x] / y_nets_star_sum
    #    else:
    #        y_finals[-1] = 0.0
    #        y_final = 1.0

    #    y_finals.append(y_final)
    
    for x in range(3):
        if y_nets_star_sum != 0.0:
            y_final1 = g * y_nets_star[2*x] / y_nets_star_sum
            y_final2 = g * y_nets_star[2*x + 1] / y_nets_star_sum
        else:
            #y_final1 = random.random()/10
            #y_final2 = random.random()/10
            y_final1 = 0.0
            y_final2 = 0.0
        y_finals.append(y_final1)
        y_finals.append(y_final2)
    
    return y_finals

def motor_to_angles(ys, sa, ea, wa, constraints = True):
    
    d_sa = g*(ys[0] - ys[1])
    d_ea = g*(ys[2] - ys[3])
    d_wa = g*(ys[4] - ys[5])
    
    new_sa = sa + d_sa
    new_ea = ea + d_ea
    new_wa = wa + d_wa
    
    if constraints:
    
        if  -np.pi > new_sa:
            new_sa = -np.pi
        elif np.pi < new_sa:
            new_sa = np.pi

        if -np.pi > new_ea:
            new_ea = -np.pi
        elif np.pi < new_ea:
            new_ea = np.pi

        if  0 > new_wa:
            new_wa = 0
        elif np.pi < new_wa:
            new_wa = np.pi
    
    
    return new_sa, new_ea, new_wa












def draw_squares(squares):

    global sim_display

    i=0
    for sq in squares:
        c = int(sq[0]*255)
        x = (i%21)*24
        y = ((i - (i%21))//21)*24 #diff function?? use that!
        if c != 0:
            pygame.draw.rect(sim_display, (c,c,c), [y-12, x-12, 24, 24])
        pygame.draw.circle(sim_display, (255,0,0), (y,x), 1)
        i = i+1
        
def draw_arm(shoulder, sa, ea, wa, colour = (255,255,255), tiles=True):

    global sim_display
    
    (shoulder, elbow, wrist, tip) = arm_forward_kinematics(shoulder, sa, ea, wa)
    
    l = l_shoulder + l_elbow + l_wrist
    
    #pygame.draw.circle(display, (70,70,70), shoulder, 1.03*l)
    #pygame.draw.circle(display, (0,0,0), shoulder, l)
    if tiles:
        squares = generate_h_vector(tip, neurons = 21)
        draw_squares(squares)
    
    pygame.draw.line(sim_display, colour, shoulder, elbow,int(l_shoulder/10))
    pygame.draw.line(sim_display, colour, elbow, wrist,int(l_elbow/10))
    pygame.draw.line(sim_display, colour, wrist, tip,int(l_wrist/10))

    R = int(l_shoulder*0.1)
    R2 = int(0.7*R)
    pygame.draw.circle(sim_display, colour, shoulder, R)
    pygame.draw.circle(sim_display, (0,0,0), shoulder, R2)
    R = int(l_elbow*0.15)
    R2 = int(0.7*R)
    pygame.draw.circle(sim_display, colour, elbow, R)
    pygame.draw.circle(sim_display, (0,0,0), elbow, R2)
    R = int(l_wrist*0.15)
    R2 = int(0.7*R)
    pygame.draw.circle(sim_display, colour, wrist, R)
    pygame.draw.circle(sim_display, (0,0,0), wrist, R2)
    R = int(l_wrist/10)
    pygame.draw.circle(sim_display, colour, tip, R)
    
    return tip
    
def menu(background = (0,0,0)):
    global sim_display

    sim_display.fill(background)
    pygame.draw.rect(sim_display, (0,0,230), [500, 0, 300, 500])
    pygame.draw.rect(sim_display, (100,100,100), [500, 0, 5, 500])
    
def slider(x, y, sx):

    global sim_display
    
    pygame.draw.rect(sim_display, (150,150,150), [x, y, 200, 5])
    pygame.draw.rect(sim_display, (255,255,255), [sx, y-10, 10, 25])
    
    if pygame.mouse.get_pressed() == (True, False, False):
        (posx, posy) = pygame.mouse.get_pos()
        #if (sx-1
        # 0 <= posx <= sx+10) and (40 <= posy <= 65) and (545 <= posx <= 745):
        if (sx-10 <= posx <= sx+10) and (y-10 <= posy <= y+15) and (545 <= posx <= 745):
            sx = posx
            #print(sx)
            #pygame.draw.rect(display, (255,255,255), [posx-5, 40, 10, 25])
    
    progress = (sx-545)/(200)
    
    return sx, progress
 

def random_movement_simulation():

    #create a game loop
    # 1. create a random movement
    # 2. update the W_pm and W_i

    global sim_display

    center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
    sa = 0
    ea = 0
    wa = 0
    # initialising pygame
    pygame.init()

    #W_PM = np.zeros((405,441))
    #W_PM = np.loadtxt('W_PM.csv', delimiter = ',') 

    sim_display = pygame.display.set_mode((WIDTH, HEIGHT))

    ################
    #choose the h you want to reach


    ###############

    while True:
        clock = pygame.time.Clock()

        # creating a loop to check events that are occurring
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        menu()
        
        _, sa, ea, wa = motor_babbling(sa, ea, wa, True)    
        
        tip = draw_arm(center,sa,ea,wa)
        
        h = generate_h_vector(tip)
        
        p = generate_p_vector(sa,ea,wa)
        
        #W_PM = update_W_PM(W_PM, p, h, epsilon = 0.01)

        # updating the display
        #pygame.display.update()
        pygame.display.flip()
        clock.tick(10)



def train_weights(iters = 10000, theta = 0.1, epsilon = 0.001, file_name = '', folder = ''):

    print("Starting the training process of W_PM and W_is for "+str(iters)+" iterations.")
    ims = []
    fig, ax = plt.subplots()

    figure, axis = plt.subplots(2, 3, figsize=(10,8))
    images = []


    (sa, ea, wa) = (0, 0, 0)
    center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))

    p_past = generate_p_vector(sa,ea, wa)

    progress = ''
    for p in range(100):
        progress = progress + '🔲'

    prog_counter = 0
        
    W_PM, W_is, rs = load_matrices(file_name= file_name, folder= folder)

    for i in range(iters):
        
        ys, sa, ea, wa = motor_babbling(sa, ea, wa, True)    
        
        (_, _, _, tip) = arm_forward_kinematics(center, sa, ea, wa)
        
        h = generate_h_vector(tip)
        
        p = generate_p_vector(sa,ea,wa)

        W_PM = update_W_PM(W_PM, p, h, epsilon=epsilon)
        
        for j in range(6):
            rs[j] = leaky_integrator_r(ys[j], p_past, rs[j])
            W_is[j] = update_W_i(W_is[j], rs[j], p, i, iters, theta = theta)
        
        p_past = p.copy()

        clear_output(wait = True)
        print(progress)
        print("The weight training is " + str(prog_counter) + '% complete')

        if i%(iters//100) == 0:
            progress =  '🔳' + progress[:-1]
            prog_counter = prog_counter + 1
            
            ax.set(xlim=(0, 405), ylim=(0, 405))
            im = ax.imshow((W_PM)**(1/100), cmap='hot', interpolation = 'nearest', animated = True)
            ims.append([im])
            
            image1 = axis[0,0].imshow((W_is[0])**(1/100), cmap='hot', interpolation = 'nearest', animated = True)
            image2 = axis[0,1].imshow((W_is[1])**(1/100), cmap='hot', interpolation = 'nearest', animated = True)
            image3 = axis[0,2].imshow((W_is[2])**(1/100), cmap='hot', interpolation = 'nearest', animated = True)
            image4 = axis[1,0].imshow((W_is[3])**(1/100), cmap='hot', interpolation = 'nearest', animated = True)
            image5 = axis[1,1].imshow((W_is[4])**(1/100), cmap='hot', interpolation = 'nearest', animated = True)
            image6 = axis[1,2].imshow((W_is[5])**(1/100), cmap='hot', interpolation = 'nearest', animated = True)
            
            images.append([image1, image2, image3, image4, image5, image6])


    print('theta: ' + str(theta))

    save_matrices(W_PM, W_is, rs, file_name = file_name, folder = folder)


        
    axis[0,0].set_title('W0')
    axis[0,1].set_title('W1')
    axis[0,2].set_title('W2')
    axis[1,0].set_title('W3')
    axis[1,1].set_title('W4')
    axis[1,2].set_title('W5')

    figure.suptitle('Sensorimotor command matrices')

    aniW_i = animation.ArtistAnimation(figure, images, interval=50, blit=True, repeat_delay=1000)
    aniW_i.save(folder+"W_i_progressionLinux.mp4")


    ax.set_title('W_PM')
    fig.suptitle('Posture memory matrix')

    aniWPM = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
    aniWPM.save(folder+"W_PM_progressionLinux.mp4")

def train_wpm(iters = 400000, epsilon = 0.001, file_name = '', folder = ''):

    print("Starting the training process of W_PM and W_is for "+str(iters)+" iterations.")
    ims = []
    fig, ax = plt.subplots()


    (sa, ea, wa) = (0, 0, 0)
    center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))

    progress = ''
    for p in range(100):
        progress = progress + '🔲'

    prog_counter = 0
        
    W_PM = np.loadtxt(folder+'W_PM'+file_name+'.csv', delimiter = ',')

    for i in range(iters):
        
        _, sa, ea, wa = motor_babbling(sa, ea, wa, True)    
        
        (_, _, _, tip) = arm_forward_kinematics(center, sa, ea, wa)
        
        h = generate_h_vector(tip)
        
        p = generate_p_vector(sa,ea,wa)

        W_PM = update_W_PM(W_PM, p, h, epsilon = epsilon)

        clear_output(wait = True)
        print(progress)
        print("The weight training is " + str(prog_counter) + '% complete')

        if i%(iters//100) == 0:
            progress =  '🔳' + progress[:-1]
            prog_counter = prog_counter + 1
            
            ax.set(xlim=(0, 405), ylim=(0, 405))
            im = ax.imshow((W_PM)**(1/100), cmap='hot', interpolation = 'nearest', animated = True)
            ims.append([im])

    np.savetxt(folder+'W_PM'+file_name+'.csv', W_PM, delimiter=',')

    ax.set_title('W_PM')
    fig.suptitle('Posture memory matrix')

    aniWPM = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
    aniWPM.save(folder+"W_PM_progressionLinux.mp4")


def check_W_PM():

    global sim_display


    W_PM = np.loadtxt('W_PM.csv', delimiter = ',')

    center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
    sa = 0
    ea = 0
    wa = 0
    # initialising pygame
    pygame.init()

    sim_display = pygame.display.set_mode((WIDTH, HEIGHT))

    h = 0

    while True:
        clock = pygame.time.Clock()
    
        # creating a loop to check events that are occurring
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        menu()
        
        (hx, hy) = h_to_space(h)
        
        p = np.argmax(W_PM[:,h])
        (sa, ea, wa) = p_to_angles(p)
        
        tip = draw_arm(center,sa,ea,wa)
        pygame.draw.rect(sim_display, (0,180,50), [hx-12, hy-12, 24, 24])
        # updating the display
        #pygame.display.update()
        pygame.display.flip()
        if h < W_PM.shape[1]-1:
            h = h + 1
        else:
            h = 0
        clock.tick(30)





def SURE_REACH_goal_space(h = 180):


    global sim_display


    W_PM, W_is, rs = load_matrices()

    center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
    sa = 0
    ea = 0
    wa = 0
    # initialising pygame
    pygame.init()

    sim_display = pygame.display.set_mode((WIDTH, HEIGHT))

    ################
    #choose the h you want to reach
    #h = 180 #random.randint(0,440)
    ###############

    h_goal = np.zeros((441,1))
    h_goal[h, 0] = 1

    (hx, hy) = h_to_space(h)



    #p_goal = np.zeros((405,1))
    #p_goal[max_row, 0] = 1
    p_goal = np.matmul(W_PM, h_goal)
    p_goal = normalize_vec(p_goal)

    max_row = np.argmax(p_goal)

    #print('p_goal: ' + str(p_goal))
    p = generate_p_vector(sa, ea, wa)

    a = [np.zeros((405,1)) for v in range(6)]
    change = 0

    while True:
        
        #if change == 200:
        #    change = 0
        #    h_goal = np.zeros((441,1))
        #    h = random.randint(0,440)
        #    h_goal[h, 0] = 1
        #    (hx, hy) = h_to_space(h)

        #    p_goal = np.matmul(W_PM, h_goal)
        #    max_row = np.argmax(p_goal)
        
        #change = change + 1
        
        clock = pygame.time.Clock()
    
        # creating a loop to check events that are occurring
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        menu()
        
        
        a = update_as(a, p_goal, W_is)
        ys = goal_motor_commands(a, p)
        
        sa,ea,wa = motor_to_angles(ys, sa, ea, wa, True)
        
        angles_goal = p_to_angles(max_row)
        draw_arm(center,angles_goal[0],angles_goal[1],angles_goal[2], colour = (70,70,70))
        
        tip = draw_arm(center,sa,ea,wa)
        
        p = generate_p_vector(sa, ea, wa)
        
        pygame.draw.rect(sim_display, (0,180,50), [hx-12, hy-12, 24, 24])
        
        
        clear_output(wait = True)
        
        print('motor commands ys: ' + str(ys))
        #print('arm angles ->')
        #print('shoulder: '+ str(sa))
        #print('elbow: '+ str(ea))
        #print('wrist: '+ str(wa))
        print('a activiation map ->')
        fig, axs = plt.subplots(1,6, figsize = (5,10))

        axs[0].imshow(np.reshape(a[0], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
        axs[1].imshow(np.reshape(a[1], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
        axs[2].imshow(np.reshape(a[2], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
        axs[3].imshow(np.reshape(a[3], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
        axs[4].imshow(np.reshape(a[4], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
        axs[5].imshow(np.reshape(a[5], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)

        plt.show()

        pygame.display.flip()
        clock.tick(5)



def movement_trace():
    global sim_display
    W_0 = np.loadtxt('W_5.csv', delimiter = ',')

    center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
    sa = 0
    ea = 0
    wa = 0
    # initialising pygame
    pygame.init()

    sim_display = pygame.display.set_mode((WIDTH, HEIGHT))

    p = 0

    while True:
        clock = pygame.time.Clock()
    
        # creating a loop to check events that are occurring
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        menu()
        
        
        dummy = W_0[:,p].copy()
        
        trees = 5
        
        step = 255//trees
        
        fade = 0
        
        rank = []
        
        for j in range(trees):
            wp = np.argmax(dummy)
            rank.append(wp)
            dummy[wp] = 0
        
        rank = rank[::-1]
        
        for i in range(trees):
            
            (wsa, wea, wwa) = p_to_angles(rank[i])

            tip_wp = draw_arm(center,wsa,wea,wwa, (fade,fade,fade), False)
            
            fade = fade+step
        
        (sa, ea, wa) = p_to_angles(p)
        
        tip_p = draw_arm(center,sa,ea,wa, (255,100,100), False)
        pygame.display.flip()
        if p < W_0.shape[1]-1:
            p = p + 1
        else:
            p = 0
        clock.tick(10)


def SURE_REACH_space_error(goal_cords, init_angles = np.array([0,0,0]), visuals = False, v = [1,1,1,1,1,1], beta = 0.172, gamma = 0.434, folder = '', file_name = ''):


    if visuals:
        global sim_display


        history = []

        W_PM, W_is, _ = load_matrices(file_name = file_name, folder= folder)

        center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
        sa = init_angles[0]
        ea = init_angles[1]
        wa = init_angles[2]
        # initialising pygame
        pygame.init()

        sim_display = pygame.display.set_mode((WIDTH, HEIGHT))

        ################
        #choose the h you want to reach
        #h = 180 #random.randint(0,440)
        ###############
        h_goal = generate_h_vector(goal_cords)



        #p_goal = np.zeros((405,1))
        #p_goal[max_row, 0] = 1
        p_goal = np.matmul(W_PM, h_goal)
        p_goal = normalize_vec(p_goal)

        max_row = np.argmax(p_goal)

        #print('p_goal: ' + str(p_goal))
        p = generate_p_vector(sa, ea, wa)

        a = [np.zeros((405,1)) for dummy in range(6)]

        for i in range(50):
            a = update_as(a, p_goal, W_is, v = v, beta = beta, gamma = gamma)

        #while True:
        for move in range(200):

            clock = pygame.time.Clock()

            # creating a loop to check events that are occurring
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    #sys.exit()

            #menu/ canvas
            menu()

            #a = update_as(a, p_goal, W_is)
            ys = goal_motor_commands(a, p)

            sa,ea,wa = motor_to_angles(ys, sa, ea, wa, True)

            angles_goal = p_to_angles(max_row)

            draw_arm(center,angles_goal[0],angles_goal[1],angles_goal[2], colour = (70,70,70))

            tip = draw_arm(center,sa,ea,wa)

            p = generate_p_vector(sa, ea, wa)

            pygame.draw.rect(sim_display, (0,180,50), [goal_cords[0]-12, goal_cords[1]-12, 24, 24])


            clear_output(wait = True)

            print('motor commands ys: ' + str(ys))
            #print('arm angles ->')
            #print('shoulder: '+ str(sa))
            #print('elbow: '+ str(ea))
            #print('wrist: '+ str(wa))
            print('a activiation map ->')
            fig, axs = plt.subplots(1,6, figsize = (5,10))

            axs[0].imshow(np.reshape(a[0], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
            axs[1].imshow(np.reshape(a[1], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
            axs[2].imshow(np.reshape(a[2], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
            axs[3].imshow(np.reshape(a[3], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
            axs[4].imshow(np.reshape(a[4], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
            axs[5].imshow(np.reshape(a[5], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)

            plt.show()

            pygame.display.flip()
            clock.tick(5)

            history.append([[tip[0], tip[1]],[sa,ea,wa]])

        pygame.quit()
        return [history, [goal_cords[0], goal_cords[1]]]

    else:

        history = []

        W_PM, W_is, _ = load_matrices(file_name = file_name, folder = folder)

        center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
        sa = init_angles[0]
        ea = init_angles[1]
        wa = init_angles[2]

        h_goal = generate_h_vector(goal_cords)

        p_goal = np.matmul(W_PM, h_goal)
        p_goal = normalize_vec(p_goal)

        p = generate_p_vector(sa, ea, wa)

        a = [np.zeros((405,1)) for dummy in range(6)]

        for i in range(50):
            a = update_as(a, p_goal, W_is, beta = beta, gamma = gamma, v = v)

        #while True:
        for move in range(200):

            ys = goal_motor_commands(a, p)

            sa,ea,wa = motor_to_angles(ys, sa, ea, wa, True)

            [_, _, _, tip] = arm_forward_kinematics(center, sa, ea, wa)

            p = generate_p_vector(sa, ea, wa)

            history.append([[tip[0], tip[1]],[sa,ea,wa]])

        return [history, [goal_cords[0], goal_cords[1]]]


def SURE_REACH_posture_error(goal_angles, init_angles = np.array([0,0,0]), visuals = False, beta = 0.172, gamma = 0.434, v = [1,1,1,1,1,1], folder = '', file_name = ''):


    if visuals:
        global sim_display


        history = []

        _, W_is, _ = load_matrices(file_name = file_name, folder= folder)

        center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
        sa = init_angles[0]
        ea = init_angles[1]
        wa = init_angles[2]
        # initialising pygame
        #sa = np.pi
        #ea = np.pi
        #wa = np.pi
        pygame.init()

        sim_display = pygame.display.set_mode((WIDTH, HEIGHT))

        ################
        #choose the h you want to reach
        #h = 180 #random.randint(0,440)
        ###############
        p_goal = generate_p_vector(goal_angles[0], goal_angles[1], goal_angles[2])

        #print('p_goal: ' + str(p_goal))
        p = generate_p_vector(sa, ea, wa)

        a = [np.zeros((405,1)) for dummy in range(6)]

        for i in range(50):
            a = update_as(a, p_goal, W_is, beta = beta, gamma = gamma, v = v)

        
        #while True:
        for move in range(200):

            clock = pygame.time.Clock()

            # creating a loop to check events that are occurring
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    #sys.exit()

            #menu/ canvas
            menu()
            a = update_as(a, p_goal, W_is, v = v, beta= beta, gamma = gamma)
            #a = update_as(a, p_goal, W_is)
            ys = goal_motor_commands(a, p)

            sa,ea,wa = motor_to_angles(ys, sa, ea, wa, True)

            #draw the desired posture to be reached
            draw_arm(center, goal_angles[0], goal_angles[1], goal_angles[2], colour = (70,70,70))

            tip = draw_arm(center,sa,ea,wa)

            p = generate_p_vector(sa, ea, wa)

            clear_output(wait = True)

            print('motor commands ys: ' + str(ys))
            #print('arm angles ->')
            #print('shoulder: '+ str(sa))
            #print('elbow: '+ str(ea))
            #print('wrist: '+ str(wa))
            print('a activiation map ->')
            fig, axs = plt.subplots(1,6, figsize = (5,10))

            axs[0].imshow(np.reshape(a[0], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
            axs[1].imshow(np.reshape(a[1], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
            axs[2].imshow(np.reshape(a[2], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
            axs[3].imshow(np.reshape(a[3], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
            axs[4].imshow(np.reshape(a[4], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)
            axs[5].imshow(np.reshape(a[5], (27,15) ), cmap='hot', interpolation = 'nearest', animated = True)

            plt.show()

            pygame.display.flip()
            clock.tick(5)

            history.append([[tip[0], tip[1]],[sa,ea,wa]])

        pygame.quit()
        return [history, [goal_angles[0], goal_angles[1], goal_angles[2]]]

    else:

        history = []

        _, W_is, _ = load_matrices(folder = folder, file_name = file_name)

        center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
        sa = init_angles[0]
        ea = init_angles[1]
        wa = init_angles[2]

        p_goal = generate_p_vector(goal_angles[0], goal_angles[1], goal_angles[2])

        p = generate_p_vector(sa, ea, wa)

        a = [np.zeros((405,1)) for v in range(6)]

        for i in range(50):
            a = update_as(a, p_goal, W_is, gamma = gamma, beta = beta, v = v)

        #while True:
        for move in range(200):

            ys = goal_motor_commands(a, p)

            sa,ea,wa = motor_to_angles(ys, sa, ea, wa, True)

            [_, _, _, tip] = arm_forward_kinematics(center, sa, ea, wa)

            p = generate_p_vector(sa, ea, wa)

            history.append([[tip[0], tip[1]],[sa,ea,wa]])

        return [history, [goal_angles[0], goal_angles[1], goal_angles[2]]]


def control(constraints = True):

    global sim_display, sx1, sx2, sx3
    # initialising pygame
    pygame.init()

    sim_display = pygame.display.set_mode((WIDTH, HEIGHT))

    while True:
    
        # creating a loop to check events that are occurring
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        menu()
        sx1, sa = slider(550,50,sx1)
        sx2, ea = slider(550,100,sx2)
        sx3, wa = slider(550,150,sx3)
        #print(progress)
        
        if constraints:
            shoulder = 2*np.pi*(sa - 0.5)
            elbow = 2*np.pi*(ea - 0.5)
            wrist = np.pi*wa
        else:
            shoulder = 2*np.pi*sa
            elbow = 2*np.pi*ea
            wrist = 2*np.pi*wa


        center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
        #pygame.draw.circle(display, (255,255,255), center, l1+l2+l3)
        
        tip = draw_arm(center, shoulder, elbow, wrist)
        
        #drawrec()
        # updating the display
        #pygame.display.flip()
        pygame.display.update()

def demo_posture(file_name = "", folder = "", gamma = 0.434, beta = 0.172, v = [1,1,1,1,1,1]):

    global sim_display, sx1, sx2, sx3
    # initialising pygame
    pygame.init()

    scur, ecur, wcur = 0,0,0

    sim_display = pygame.display.set_mode((WIDTH, HEIGHT))

    _, W_is, _ = load_matrices(file_name = file_name, folder= folder)

    a = [np.zeros((405,1)) for dummy in range(6)]
    

    while True:
    
        # creating a loop to check events that are occurring
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        menu()
        sx1, sa = slider(550,50,sx1)
        sx2, ea = slider(550,100,sx2)
        sx3, wa = slider(550,150,sx3)
        #print(progress)
        
       
        goal_shoulder = 2*np.pi*(sa - 0.5)
        goal_elbow = 2*np.pi*(ea - 0.5)
        goal_wrist = np.pi*wa
    


        center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
        #pygame.draw.circle(display, (255,255,255), center, l1+l2+l3)
        
        _ = draw_arm(center, goal_shoulder, goal_elbow, goal_wrist, colour=(100,100,100))

        tip = draw_arm(center, scur, ecur, wcur)

        p_goal = generate_p_vector(goal_shoulder, goal_elbow, goal_wrist)

        p = generate_p_vector(scur, ecur, wcur)

        a = update_as(a, p_goal, W_is, gamma = gamma, beta = beta, v = v)

        ys = goal_motor_commands(a, p)

        scur,ecur,wcur = motor_to_angles(ys, scur, ecur, wcur, True)

        #[_, _, _, tip] = arm_forward_kinematics(center, sa, ea, wa)

        #p = generate_p_vector(sa, ea, wa)
        
        #drawrec()
        # updating the display
        #pygame.display.flip()
        pygame.display.update()


def reduced_joint_mobility(file_name = "", folder = "", gamma = 0.434, beta = 0.172, v = [1,1,1,1,1,1]):

    global sim_display
    # initialising pygame
    pygame.init()
    v2 = [1,1, 1, 1, 1,1]
    s1, e1, w1 = -np.pi, -np.pi/1.5, 1
    s2, e2, w2 = -np.pi, -np.pi/1.5, 1
    #sg, eg, wg = np.pi/2, np.pi/6, 0
    #goal_cords = np.array([300,20])
    goal_cords = np.array([150,150])
    sim_display = pygame.display.set_mode((WIDTH, HEIGHT))

    W_PM, W_is, _ = load_matrices(file_name = file_name, folder= folder)

    a1 = [np.zeros((405,1)) for dummy in range(6)]
    a2 = [np.zeros((405,1)) for dummy in range(6)]

    pics = []
    pics.append([[s1,e1,w1], [s2,e2,w2]])
    flash = 0

    run = True
    while run:
    
        # creating a loop to check events that are occurring
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                #pygame.quit()
                #sys.exit()
        
        menu(background=(255,255,255))
        #print(progress)
    
        center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
        #pygame.draw.circle(display, (255,255,255), center, l1+l2+l3)
        
        flash = flash + 1
        if flash == 25:
            pics.append([[s1,e1,w1], [s2,e2,w2]])
            flash = 0

        pygame.draw.circle(sim_display, (0,0,100), goal_cords, 20)
        _ = draw_arm(center, s2, e2, w2, colour=(0,0,0), tiles=False)

        _ = draw_arm(center, s1, e1, w1, colour=(255,50,50), tiles=False)

        h_goal = generate_h_vector(goal_cords)
        p_goal = np.matmul(W_PM, h_goal)
        p_goal = normalize_vec(p_goal)

        p1 = generate_p_vector(s1, e1, w1)
        p2 = generate_p_vector(s2, e2, w2)


        a1 = update_as(a1, p_goal, W_is, gamma = gamma, beta = beta, v = v)
        a2 = update_as(a2, p_goal, W_is, gamma = gamma, beta = beta, v = v2)

        ys1 = goal_motor_commands(a1, p1)
        ys2 = goal_motor_commands(a2, p2)


        s1,e1,w1 = motor_to_angles(ys1, s1, e1, w1, True)
        s2,e2,w2 = motor_to_angles(ys2, s2, e2, w2, True)

        #[_, _, _, tip] = arm_forward_kinematics(center, sa, ea, wa)

        #p = generate_p_vector(sa, ea, wa)
        
        #drawrec()
        # updating the display
        pygame.display.flip()

    pygame.quit()
    return pics
        #pygame.display.update()


def obstacle_perception(x1,x2,y1,y2):
    
    if x1 >= 0 and y1 >= 0 and x2 <= 21 and y2 <= 21:
        
        W = np.zeros((21,21))
        W[x1:x2, y1:y2] = np.ones((x2-x1, y2-y1))
        h = W.reshape((441,1), order = 'F')
        
    else:
        return "wrong coordinates out of workspace"
    
    return h

def obstacle_avoidence(x1,x2, y1,y2, file_name = "", folder = "", gamma = 0.434, beta = 0.172, v = [1,1,1,1,1,1]):
    global sim_display
    # initialising pygame
    pygame.init()
    s1, e1, w1 = 0, 0, 0
    s2, e2, w2 = 0, 0, 0
    #sg, eg, wg = np.pi/2, np.pi/6, 0
    #goal_cords = np.array([300,20])
    goal_cords = np.array([50,200])
    sim_display = pygame.display.set_mode((WIDTH, HEIGHT))

    W_PM, W_is, _ = load_matrices(file_name = file_name, folder= folder)
    #W_PM_obs, W_is_obs, _ = load_matrices(file_name = file_name, folder= folder)


    h_deac = obstacle_perception(y1,y2, x1,x2)
    p_deac = normalize_vec(np.matmul(W_PM, h_deac))
    
    #p_bin = np.zeros((405,1))
    #for i in range(405):
    #    if p_deac[i,0] < 0.01:
    #        p_deac[i,0] = 0.0

    a = [np.zeros((405,1)) for dummy in range(6)]
    a_obs = [np.zeros((405,1)) for dummy in range(6)]

    pics = []
    pics.append([[s1,e1,w1], [s2,e2,w2]])
    flash = 0

    run = True
    while run:
    
        # creating a loop to check events that are occurring
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                #pygame.quit()
                #sys.exit()
        
        menu(background=(255,255,255))
        #print(progress)
    
        center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
        #pygame.draw.circle(display, (255,255,255), center, l1+l2+l3)
        
        flash = flash + 1
        if flash == 25:
            pics.append([[s1,e1,w1], [s2,e2,w2]])
            flash = 0

        pygame.draw.rect(sim_display, (200,0,0), pygame.Rect(24*x1, 24*y1, 24*(x2-x1), 24*(y2-y1)))

        pygame.draw.circle(sim_display, (0,0,100), goal_cords, 20)
        _ = draw_arm(center, s2, e2, w2, colour=(0,0,0), tiles=False)

        _ = draw_arm(center, s1, e1, w1, colour=(255,50,50), tiles=False)

        h_goal = generate_h_vector(goal_cords)
        p_goal = np.matmul(W_PM, h_goal)
        p_goal = normalize_vec(p_goal)

        p = generate_p_vector(s1, e1, w1)
        p_obs = generate_p_vector(s2, e2, w2)

        '''for i in range(405):
            for j in range(6):
                if p_deac[i,0] > 0.01: # and a_obs[j][i,0] > 0.01:
                    #a_obs[j][i,0] = 0.0
                    p_obs[i,0] = 0.0'''

        a = update_as(a, p_goal, W_is, gamma = gamma, beta = beta, v = v)
        a_obs = update_as_obs(a_obs, p_goal, p_deac, W_is, gamma = gamma, beta = beta, v = v) #convolute with p_deac?
        
        '''for i in range(405):
            for j in range(6):
                if p_deac[i,0] > 0.01: # and a_obs[j][i,0] > 0.01:
                    #a_obs[j][i,0] = 0.0
                    p_obs[i,0] = 0.0'''

        #for i in range(6):
            #a_obs[i] = normalize_vec(a_obs[i])

        ys = goal_motor_commands(a, p)
        ys_obs = goal_motor_commands(a_obs, p_obs)


        s1,e1,w1 = motor_to_angles(ys, s1, e1, w1, True)
        s2,e2,w2 = motor_to_angles(ys_obs, s2, e2, w2, True)

        #[_, _, _, tip] = arm_forward_kinematics(center, sa, ea, wa)

        #p = generate_p_vector(sa, ea, wa)
        
        #drawrec()
        # updating the display
        pygame.display.flip()

    pygame.quit()
    return pics
        #pygame.display.update()


def deac_trace(p_deac, x1, x2, y1, y2, trees = 6):
    global sim_display
    center = (int(WIDTH_ARM/2),int(HEIGHT_ARM/2))
    # initialising pygame
    pygame.init()

    sim_display = pygame.display.set_mode((WIDTH, HEIGHT))

    while True:
        clock = pygame.time.Clock()
    
        # creating a loop to check events that are occurring
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        menu()
        pygame.draw.rect(sim_display, (200,0,0), pygame.Rect(24*x1, 24*y1, 24*(x2-x1), 24*(y2-y1)))

        dummy = p_deac.copy()
        
        step = 255//trees
        
        fade = 0
        
        rank = []
        
        for j in range(trees):
            wp = np.argmax(dummy)
            rank.append(wp)
            dummy[wp] = 0
        
        rank = rank[::-1]
        
        for i in range(trees):
            
            (wsa, wea, wwa) = p_to_angles(rank[i])

            #_ = draw_arm(center,wsa,wea,wwa, (fade,fade,fade), False)
            _ = draw_arm(center,wsa,wea,wwa, (255,255,255), False)

            fade = fade+step
        
        pygame.display.update()
        clock.tick(10)

