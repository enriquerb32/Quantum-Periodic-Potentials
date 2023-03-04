# Solution of Schrodinger equation in 1D
# Bound states
#
#  Discretize 1D equation in a line
#  Set potential in the middle of the line
#  Diagonalize sparse matrix for a few eigenstates

import numpy as np
import scipy.linalg as spy
import matplotlib.pyplot as plt


#####################
#   TABLE OF CONTENTS
#
#   I. MAIN FUNCTIONS
#   II. Plot functions
#   III.  Potential functions
#   IV.   Auxilary functions
#   V.  Solution functions (check in case of despair).


#############################
#  I.  MAIN  FUNCTIONS

#   1) HAM  :  computes hamiltonian matrix for a given potential
#   2) TRANS:  computes T(E) for a given potential

###########################
def hamk(xmax,N,potfun,param,mass,kL):
    '''

    Description
    discretize kinetic operator -h^2/2m (d^2/dx^2) in the line (-xmax, xmax) 
    INPUTS
    *xmax is a real number. It goes in AA
    *N  (integer) is the number of points in the grid.
    Therefore, there are N-1 segments
    * potfun is a function f(x,param), to be provided externally
    * param is a set of parameters, 
    
    -hbar^2/2m Phi``= (hbar^2/2m delta^2)  ( + 2 Phi(n) - Phi(n+1)-Phi(n+1))
    *ADDS potential in diagonal

    OUTPUT: A dimension N numpy array (matrix) with the discretized Hamiltonian
    in meV
(    '''
    from scipy.sparse import dia_matrix
    
    dx=2.*xmax/float(N-1) #size of step
    
    hbarsquareovertwom=calchbarsquareovertwom() # approximately in meV AA^2
    
    epsilon=hbarsquareovertwom/(dx*dx)  # energy in meV
    epsilon=epsilon/mass # if mass=1,  we have electrons
    
    mat=np.array([[0.0j for i in range(N)] for j in range(N)])
    #dia_matrix((N,N),dtype=np.float).toarray()  # this creates an empty matrix
    
   
#  We now fill the diagonal
    x=-xmax
    for i in np.arange(N):
        mat[i,i]=+2.*epsilon+potfun(x,param)
        x=x+dx

#  We now fill the positive co-diagonal the (i,i+1) terms
    for i in np.arange(N-1):
        mat[i,i+1]=-epsilon

#  We now fill the co-diagonal the (i+1,i) terms
    for i in np.arange(N-1):
        mat[i+1,i]=mat[i,i+1]

    if choice2:
        mat[0,N-1]=np.exp(kL*1j)*(-epsilon)
        mat[N-1,0]=np.exp(-kL*1j)*(-epsilon)   
        
    return mat # in meV


###########################
def ham(xmax,N,potfun,param,mass):
    '''

    Description
    discretize kinetic operator -h^2/2m (d^2/dx^2) in the line (-xmax, xmax) 
    INPUTS
    *xmax is a real number. It goes in AA
    *N  (integer) is the number of points in the grid.
    Therefore, there are N-1 segments
    * potfun is a function f(x,param), to be provided externally
    * param is a set of parameters, 
    
    -hbar^2/2m Phi``= (hbar^2/2m delta^2)  ( + 2 Phi(n) - Phi(n+1)-Phi(n+1))
    *ADDS potential in diagonal

    OUTPUT: A dimension N numpy array (matrix) with the discretized Hamiltonian
    in meV
(    '''
    from scipy.sparse import dia_matrix
    
    dx=2.*xmax/float(N-1) #size of step
    
    hbarsquareovertwom=calchbarsquareovertwom() # approximately in meV AA^2
    
    epsilon=hbarsquareovertwom/(dx*dx)  # energy in meV
    epsilon=epsilon/mass # if mass=1,  we have electrons
    
    mat=np.array([[0.0j for i in range(N)] for j in range(N)])
    #dia_matrix((N,N),dtype=np.float).toarray()  # this creates an empty matrix
    
    
#  We now fill the diagonal
    x=-xmax
    for i in np.arange(N):
        mat[i,i]=+2.*epsilon+potfun(x,param)  #
        x=x+dx

#  We now fill the positive co-diagonal the (i,i+1) terms
    for i in np.arange(N-1):
        mat[i,i+1]=-epsilon

#  We now fill the co-diagonal the (+1,i) terms
    for i in np.arange(N-1):
        mat[i+1,i]=mat[i,i+1]

    return mat # in meV

def bands1D(xmax,N,potfun,param,mass,dkL):
    """ Compute the 1D bands.
		ham is a FUNCTION of a phase."""

    klist=np.arange(0,2*np.pi,dkL)
    bands=[]
    for k in klist:
        mat=hamk(xmax,N,potfun,param,mass,k)
        #mat=ham(xmax,N,potfun,param,mass)
        eig=np.linalg.eigvalsh(mat)
        eigreal=eig.real
        bands.append(eigreal) 
	

    return bands

def plotbands1D(xmax,N,potfun,param,mass,dk,numberofbands):
    global figurenum
    kmax=2*np.pi
    print('kmax=',kmax)
    print(dk)
    klist=np.arange(0,2*np.pi,dk)
    print('len llist=',len(klist))
    bands=bands1D(xmax,N,potfun,param,mass,dk)
    
    plt.figure(figurenum)
    figurenum+=1

    plt.ion()
	
    for n in range(numberofbands):
        bb=[]
        for i in range(len(bands)):
            bb.append(bands[i][n])
        plt.plot(klist,bb,'o', label=" banda {}".format(n+1))
    plt.legend(loc="best")
    print(len(klist),len(bb))
    plt.title("Bandas") 
    plt.xlabel('ka(2*pi)')
    plt.ylabel('Bands')
    plt.xlim([0,2*np.pi])
    plt.grid()
    plt.draw()
    plt.show()

    return


###########################################################################
#  II.  PLOT FUNCTIONS

#   1)  plotstates  (to plot bound states, wave functions and their potential
#   2)  plotT:  plots T(E)
#   3) Plotpot: to plot the potential only.


def plotstates(mat,kval,xmax,N,potfun,param):
    '''
    INPUTS:
    * mat is a  hamiltonian matrix, obtained with HAM.
    * kval is the number of states that are going to be plotted
    * xmax and N define the grid (same as in HAM to define mat)
    * potfun: pofun(x,param) is a used-defined function that defines the potential

    OUTPUT
    a figure showing V(x), the energy levels and the wave functions.
    
    '''

    global figurenum

    dx=2.*xmax/float(N-1)
    #xlist=np.arange(-xmax,xmax+dx,dx)
    potlist=[]
    xlist=[]
    x=-xmax
    for k in range(N):
        xlist.append(x)
        potlist.append(potfun(x,param))
        x=x+dx

    plt.plot(xlist,potlist, color="k", linewidth=1.5, label="Potencial" )
    ener,wave=spy.eigh(mat)
    
    enerlist=[]
    wavelist=[]
    for i in range(kval):
        wavelist.append(wave[:,i])
        plt.plot(xlist,ener[i]+500*wavelist[i], label = "Estado {}"
                 .format(i+1))
        subener=[]
        for x in xlist:
            subener.append(ener[i])
            
        enerlist.append(subener)  # flat line with height ener
        plt.plot(xlist,enerlist[i],linestyle="dotted")
    
    if choice == 1:
        pot1 = "del pozo infinito"
    elif choice == 2:
        pot1 = "del potencial armónico"
    elif choice == 3:
        pot1 = "de la barrera doble"
    elif choice == 4:
        pot1 = "de {} barreras".format(Nbarrier)
    elif choice == 5 :
        pot1 = "del Delta de Dirac"
        
    plt.legend(loc="lower left")   
    plt.title("Energía de los estados {}".format(pot1)) 
    plt.xlabel('x')
    plt.ylabel('Energy (meV)')
    plt.grid()
    #plt.ylim([ymin,ener[kval]+0.5])
    plt.draw()
    plt.show()

    return
#############################
################################################
#####################
###############################################
#############################
#############################
#############################
#############################
#############################


def plotpot(xmax,N,potfun,param):
    ''' this function plots a potential in given by potfun(x,param), where
param can be one or many parameters  in the grid defined by xmax and N

    Example
    plotpot(40,100,doublebarrier,(-20,-10,10,20,100,100))
    plots the double barrier potential.
    '''
           
    potlist=np.array([0. for i in range(N)])
    xlist=np.array([0. for i in range(N)])
    
    dx=2*xmax/float(N-1)
    x=-xmax

                   
    for i in np.arange(N):
        pot=potfun(x,param)
        potlist[i]=pot
        xlist[i]=x
        x=x+dx
    plt.plot(xlist,potlist, color="k", linewidth=1.5)
#    plt.plot(xlist,wave.imag,'o')
#    plt.plot(xlist,potlist)
#    print "xmin=",x
#    print xlist
    plt.xlabel('x(AA)')
    plt.ylabel('Potential (meV)')
    plt.draw()
    plt.show()                 
    
    return

##########################################################
#  III.  Potenials
#   1.  harmonic: Harmonic potential
#   2.  Well:  square well/barrier
#   3.  Double barrier
#   4.  Multiple barrier

#############################
def harmonic(x,param):
    '''
    INPUtS:  k in meV**AA^-2,  x in AA
    OUTPUT: harmonic potential in meV'''
    k=param[0]
    x0=param[1]
    pot=(0.5*k*(x-x0)**2)
    return pot

#############################
def well(x,param):
    '''
    param=(x0,L,V0):

    this function returns V0 if x0<x<L
    '''
    x0=param[0]
    L=param[1]
    V0=param[2]
    if(x>x0 and x<(x0+L)):
        pot=V0
    else:
        pot=0

    return pot

##########################
def doublebarrier(x,param):
#def doublebarrier(x,x0,x1,x2,x3, V1,V2):

    ''' This function returns a piecewise function of x, that takes the following values:
       if x<x0    0
       if x0<x<x1:  V1
       if x1<x<x2:  0
       if x2<x<x3:  V2
       if x>x3  0
    '''
    
    x0=param[0]
    x1=param[1]
    x2=param[2]
    x3=param[3]
    V1=param[4]
    V2=param[5]
    pot=0.
    
    if (x<x0):
        pot=0.
        
    if(x0<x<x1):
        pot=V1
        
    if(x1<x<x2):
        pot=0.
        
    if(x2<x<x3):
        pot=V2
      
    return pot

##############################
def multbarrier(x,param):
    ''' This function returns a piecewise function of x, that generats 
    N barrierrs pot=V, of width w1,  separated by (N-1) regions of width w2 and V=0takes the following values:
      
    '''
    
    pot=0.
    N=param[0]
    x0=param[1]
    w1=param[2]
    w2=param[3]
    V=param[4]
    
    if (x<x0):
        pot=0.
   
    for i in range(N):
        xmin=x0+i*(w1+w2)
        xmax=xmin+w1
        if(xmin<x<xmax):
            pot=V
        if(xmax<x<xmax+w2):
            pot=0.
            
    return pot

##########################################################
#  IV AUXILIARY FUNCTIONS

#   1.  calchbarsquareovertwom
#   2.  phase
#   3.  transquare
#   4.  getener (provides eigenvalues for an input matrix)
#   5.  ladder operator 


def calchbarsquareovertwom():
    ''' This function computes hbar^2/2 m, where m is the electron mass
    and returns its value in meV AA*2
    '''
    hbar=1.054e-34 #J.s
#    hbar=6.5821191e-15 # eV*s
    mass=9.10938356e-31 # Kg
    
    c=0.5*hbar*hbar/mass # this goes in Joule*Joule*s*s/Kg= Joule m^2
#   Joule= kg*meter^2* s^-2
    c=c*1e20/1.602e-19 # this goes in eV AA^2
    
    return c*1e3 # meV AA*2


def phase(alpha):
    return np.cos(alpha)+1j*np.sin(alpha)


def getener(mat,kval):
    ''' returns the first kval eigenvalues of kmat
    '''
    ener=spy.eigvalsh(mat)
    ener0=[]
    for i in range(kval):
        ener0.append(ener[i])
    return ener0


##########################################################
#  V SOLUTION FUNCTIONS
#   1. comparesquare (calculates T(E) using analytical formula and trans.
#   2. compHW (compares HW potential, analytically and numerically
#   3. HW. computes energy levels or hard wall, using analytical formulas
#   4. waveHW same than 3. but for the wave function.
#   5. comphar  :compares energy levels for harmonic oscillator (numeric and analytical)
#   6. hbarw:  computes hbar*w for a given k and m
#   

#################  November 17 2018
def compdouble(R,T,alpha):

    T2=T*T
    T2=T2/(T2+4*R*np.sin(alpha)**2.)
    return T2

################ November 6 2019
def deltaener(V0,a,mass):
    ''' this function calculates bound state energy for Dirac bound state 1D
        V0 is in meV
        a is in AA
        mass=1 is electrons mass

        E=-m(a V0)^2/2 hbar^2= (a V0)^2/ (hbar^2/2m)
        
    '''
    num=mass*(V0*a)**2.
    den=4*calchbarsquareovertwom()
    return -num/den
    

########Valores
mass=1
xmax=60 ### xmax no puede ser menor que la anchura del potencial.
N=200### N  (integer) is the number of points in the grid. o tb numero de divisones
V0=50###### Valor del potencial meV
dk=0.05 #delta de k
kL=0.1
x0=10   #x0= where is starting my potential
L=10


choice = int(input('Menús a elegir: Pozo infinito (1), Armónico (2),Barrera doble (3), Multibarrera (4) y Delta (5): \n'))
                   
choice2 = str(input('¿Condiciones periódicas? (s/n) \n'))

Kval = int(input('Número de estados: \n')) 

numberofbands= int(input('Número de bandas: \n'))

if choice2 == "s":
    choice2 = True
else:
    choice2 = False
    
if choice == 1:
    #WELL
    potfun=well
    if choice2:
        x0 = 0
        xmax=60
        L = 10
        V0=-60
    if not choice2:
        x0 = 0
        xmax=60
        L = 10
        V0=-60
    param=(x0,L,V0)   
    # param=(x0,L,V0):
    #x0= where is starting my potential
    #L= width pf my potential
    #V0=value of the well
      
if choice == 2:
    #HARMONIC
    if choice2:
        x0 = 0
        k = 6
        xmax = 100
    if not choice2:
        xmax = 10
        k = 6
        xmax = 100
    potfun=harmonic
    param=(0.001,x0)#valor dentro la zona de brilloin
    #param=(k, x0)
    #pot=0.5*k*x*x+x0
    

if choice == 3:
    #####DOBLE BARRIER
    potfun=doublebarrier
    # V!
    #  |      ___V0                 
    #  |     |   |                 ___V1  
    #  |     |   |                |   |
    #  |     |   |                |   |
    #  ______x0 x1_______________x2    x3
    if choice2:
       V1=10
       V0=10
       x1=15
       x2=25
       x3=30
       x0=0
       xmax = 100
    if not choice2:
       V1=10
       V0=10
       x1=15
       x2=25
       x3=30
       x0=0
       xmax = 100
    param=(x0,x1,x2,x3, V0,V1)
    #doublebarrier(x,x0,x1,x2,x3, V1,V2):
        
if choice == 4:
    #MULTIBARRIER
    potfun=multbarrier
    #N barrierrs pot=V, of width w1,  separated by (N-1) regions of width w2 and V=0takes the following values
    #param=(Nbarrier,x0,w1anchobarr,w2ancho0,V0)
    Nbarrier= int(input("Número de barreras: \n"))
    
    if choice2:
        xmax = 60
        x0=0
        w1=2
        w2=2
        V0 = 50
    if not choice2:
        xmax = 60
        x0=0
        w1=2
        w2=2
        V0 = 50
    param=(Nbarrier,x0,w1,w2,V0)

if choice == 5:
    #DIRACDELTA
    V0=1e6###### Valor del potencial meV
    L=1
    potfun=well
    if choice2:
        x0 = 0
        xmax = 60
    if not choice2:
        x0 = 0
        xmax = 60
    param=(x0,L,V0)   
    # param=(x0,L,V0):
    #x0= where is starting my potential
    #L= width pf my potential
    #V0=value of the well


##### To plot we need
#Kval=4 #    * kval is the number of states that are going to be plotted
#mat=ham(xmax,N,potfun,param,mass)
mat=hamk(xmax,N,potfun,param,mass,kL)

figurenum=1
plotstates(mat,Kval,xmax,N,potfun,param)

figurenum=2
plotbands1D(xmax,N,potfun,param,mass,dk,numberofbands)
