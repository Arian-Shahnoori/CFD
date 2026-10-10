print(''' 
Here are the coomands:
'1' = verification for GS solver: plots numerical solution of GS for Mesh1
'2' = plots convergence history of GS solver for all meshes and prints convergence CPU time and iteration
'3' = plots error norm vs. #CVs to colclude order of accuracy 
'4' = plots error norm vs. #CVs to colclude order of accuracy for the manipulated version
'5' = plots convergence history and prints convrgence CPU time and iteration for point Jacobi, Gauss-Seidel and SORs with w=1.1, 1.3, 1.5
'6' = verification for Line SOR solver: plots numerical solution of Line SOR for Mesh4 with w=1.1
'7' = plots convergence history and prints convrgence CPU time and iteration for Line SOR and SOR with w=1.1, 1.3, 1.5 for mesh4
'8' = verification for Implicit solver: plots numerical solution of Implicit method for Mesh3
'9' = prints convergence CPU time for: SOR with w = 1.3, 1.5 of mesh 1 2 3; Line SOR with w = 1.3, 1.5 of mesh 1 2 3 4; Implicit method of mesh 1 2 3 4
''')
command = input('Enter your command: >> ')

import numpy as np
import math
from matplotlib import pyplot as plt
import time

M1dimension = 10
M2dimension = 20
M3dimension = 30
M4dimension = 40
meshlist = [M1dimension,M2dimension,M3dimension,M4dimension]
#initial T ----------------------------------------------------------------------------------------------------
def initialT(dimension):
    T = np.zeros((dimension+2,dimension+2))
    for i in range(1,dimension+1):
        T[i,dimension+1] = 2*np.sin(np.pi*(1/dimension)*(i-0.5)) 
    return T

#Mean of exact value for a cell -------------------------------------------------------------------------------
def MeanExactSolutionofCVij(i,j,dimension):
    a = (1/dimension) * i
    b = (1/dimension) * (i+1)
    c = (1/dimension) * j
    d = (1/dimension) * (j+1)
    C = (1/math.pi) * (math.cos(math.pi*a) - math.cos(math.pi*b))
    D = (1/math.pi) * (math.cosh(math.pi*d) - math.cosh(math.pi*c))
    A = (C*D) / (math.sinh(math.pi))
    MeanExactValue = (dimension**2) * A
    return MeanExactValue

#--------------------------------------------------------------------------------------------------------------
def NormofResidual(T,dimension,Manipulation):
    #calculating Residual or Flux integral---------------------------------------------------------
    RL2summation = 0
    for j in range(1,dimension+1):
        for i in range(1,dimension+1):
            if Manipulation == True:
                if i == dimension//2+1  and j == dimension :
                    RL2summation += 0
                else:
                    RL2summation += ((dimension**2) * abs(T[i+1,j]+T[i-1,j]+T[i,j+1]+T[i,j-1]-4*T[i,j]))**2
            else:
                RL2summation += ((dimension**2) * abs(T[i+1,j]+T[i-1,j]+T[i,j+1]+T[i,j-1]-4*T[i,j]))**2
    #Norm of Residual (used for convergence condition)---------------------------------------------
    L2 = (RL2summation / (dimension**2))**0.5
    return L2

#--------------------------------------------------------------------------------------------------------------
def ErrorNorms(T,dimension): #(used for showing order of accuracy)
    EL1sum = 0
    EL2sum = 0
    maxerror = -1
    for j in range(1,dimension+1):
        for i in range(1,dimension+1):
            CVerror = abs(MeanExactSolutionofCVij(i-1,j-1,dimension)-T[i,j])
            if CVerror > maxerror:
                maxerror = CVerror
            EL1sum += CVerror
            EL2sum += CVerror**2
    EL1 = EL1sum / (dimension**2)
    EL2 = (EL2sum / (dimension**2))**0.5
    ELinf = maxerror
    return EL1, EL2, maxerror

#--------------------------------------------------------------------------------------------------------------
def PJacobi(T,dimension,convergencelimit):
    iteration = 0
    L2list = []
    iterationlist = []
    timelist = []
    start_time = time.time()
    #-------------------------------------------------------------
    L2 = 1
    while L2 > convergencelimit:
        formerT = T
        iteration += 1
        #updating domain nodes-------------------------------------------------------------
        for j in range(1,dimension+1):
            for i in range(1,dimension+1):
                T[i,j] = (formerT[i+1,j]+formerT[i-1,j]+formerT[i,j+1]+formerT[i,j-1]) / 4

        #updating ghost cells' nodes-------------------------------------------------------
        for j in range(1,dimension+1):
            T[0,j] = -T[1,j]
            T[dimension+1,j] = -T[dimension,j]
        for i in range(1,dimension+1):
            T[i,0] = -T[i,1]
            T[i,dimension+1] = 2*np.sin(np.pi*(1/dimension)*(i-0.5)) - T[i,dimension]

        #caculating norm of residual-------------------------------------------------------
        L2 = NormofResidual(T,dimension,False)
        #constructing aquirements for number5----------------------------------------
        iteration_time = time.time()
        L2list.append(L2)
        iterationlist.append(iteration)
        timelist.append(iteration_time - start_time)
    convergence_iteration = iteration
    convergence_time = iteration_time - start_time
    #--------------------------------------------------------------------------------
    return T, L2list, iterationlist, timelist, convergence_iteration, convergence_time


#Gauss-Seidel solver-------------------------------------------------------------------------------------------
def GS(T,dimension,convergencelimit,Manipulation):
    #aquirements for number5--------------------------------------
    iteration = 0
    L2list = []
    iterationlist = []
    timelist = []
    start_time = time.time()
    #-------------------------------------------------------------
    L2 = 1
    while L2 > convergencelimit:
        iteration += 1
        #updating domain nodes-------------------------------------------------------------
        for j in range(1,dimension+1):
            for i in range(1,dimension+1):
                if Manipulation == True:
                    if i == dimension//2+1  and j == dimension :
                        T[i,j] = 2*np.sin(np.pi*(1/dimension)*(i-0.5))
                    else:
                        T[i,j] = (T[i+1,j]+T[i-1,j]+T[i,j+1]+T[i,j-1]) / 4
                else:
                    T[i,j] = (T[i+1,j]+T[i-1,j]+T[i,j+1]+T[i,j-1]) / 4

        #updating ghost cells' nodes-------------------------------------------------------
        for j in range(1,dimension+1):
            T[0,j] = -T[1,j]
            T[dimension+1,j] = -T[dimension,j]
        for i in range(1,dimension+1):
            T[i,0] = -T[i,1]
            T[i,dimension+1] = 2*np.sin(np.pi*(1/dimension)*(i-0.5)) - T[i,dimension]

        #caculating norm of residual-------------------------------------------------------
        L2 = NormofResidual(T,dimension,Manipulation)
        #constructing aquirements for number5----------------------------------------
        iteration_time = time.time()
        L2list.append(L2)
        iterationlist.append(iteration)
        timelist.append(iteration_time - start_time)
    convergence_iteration = iteration
    convergence_time = iteration_time - start_time
    #--------------------------------------------------------------------------------
    return T, L2list, iterationlist, timelist, convergence_iteration, convergence_time

#--------------------------------------------------------------------------------------------------------------
def SOR(T,dimension,convergencelimit,w):
    #aquirements for number5-------------------------------------
    iteration = 0
    L2list = []
    iterationlist = []
    timelist = []
    start_time = time.time()
    #-------------------------------------------------------------
    L2 = 1
    while L2 > convergencelimit:
        iteration += 1
        #updating domain nodes-------------------------------------------------------------
        for j in range(1,dimension+1):
            for i in range(1,dimension+1):
                deltaT = (T[i+1][j]+T[i-1][j]+T[i][j+1]+T[i][j-1])/4 - T[i][j]# Eq.p.16; simplified for equal length scales of x & y direction                                  
                T[i][j] = T[i][j] + w*deltaT

        #updating ghost cells' nodes-------------------------------------------------------
        for j in range(1,dimension+1):
            T[0,j] = -T[1,j]
            T[dimension+1,j] = -T[dimension,j]
        for i in range(1,dimension+1):
            T[i,0] = -T[i,1]
            T[i,dimension+1] = 2*np.sin(np.pi*(1/dimension)*(i-0.5)) - T[i,dimension]

        #caculating norm of residual-------------------------------------------------------
        L2 = NormofResidual(T,dimension,False)
        #constructing aquirements for number5----------------------------------------
        iteration_time = time.time()
        L2list.append(L2)
        iterationlist.append(iteration)
        timelist.append(iteration_time - start_time)
    convergence_iteration = iteration
    convergence_time = iteration_time - start_time
    #--------------------------------------------------------------------------------
    return T, L2list, iterationlist, timelist, convergence_iteration, convergence_time

#--------------------------------------------------------------------------------------------------------------
def GaussianElimination(A,d):
    A = np.array(A)
    d = np.array(d)
    A_index = np.shape(A)
    A_row = A_index[0]
    A_column = A_index[1]
    # Augmented matrix construction-------------------------------------------------------------
    Aug = np.zeros((A_row,A_column+1))
    for i in range(A_row):
        for j in range(A_column+1):
            if j == A_column:
                Aug[i,j] = d[i]
            else:
                Aug[i,j] = A[i,j]         
    # ------------------------------------------------------------------------------------------
    # Triangular top matrix construction--------------------------------------------------------
    for i in range(A_row-1):
        for k in range(i+1,A_row):
            # concerning Aug[i,i] being zero:------------------------
            while Aug[i,i] == 0:
                for k in range(i+1,A_row):
                    e = Aug[i,:]
                    Aug[i,:] = Aug[k,:]
                    Aug[k,:] = e
            #--------------------------------------------------------
            l = -Aug[k,i]/Aug[i,i]
            Aug[k,:] = Aug[k,:] + l*Aug[i,:]
    # ------------------------------------------------------------------------------------------
    # Extracting X vector-----------------------------------------------------------------------
    X = np.zeros((A_row,1))
    for i in range(1,A_row+1):
        sum = 0
        if i == 1:
            X[A_row-i] = Aug[A_row-i,A_column] / Aug[A_row-i,A_row-i]
        else:
            for j in range(1,i):
                sum += (Aug[A_row-i,A_row-j] * X[A_row-j])
            X[A_row-i] = (Aug[A_row-i,A_column] - sum) / Aug[A_row-i,A_row-i]
    #-------------------------------------------------------------------------------------------
    return X

#--------------------------------------------------------------------------------------------------------------
def LineSOR(dimension,w):
    T = np.zeros((dimension+2,dimension+2)) # including ghost cells
    
    def upperbondaryT(i):
        x = (1/dimension)*(i-0.5)
        T = np.sin(np.pi*x)
        return T
    for i in range(1,dimension+1):
        T[i,dimension+1] = 2*upperbondaryT(i) # initial value of upper ghost cells
    
    iteration = 0
    L2list = []
    iterationlist = []
    timelist = []
    start_time = time.time()
    L2 = 1
    while L2 > 10**(-12):
        iteration += 1
        for i in range(1,dimension+1):
            A = np.zeros((dimension,dimension))
            b1 = np.transpose(T[i+1,1:dimension+1])
            b2 = np.transpose(T[i-1,1:dimension+1])
            b3 = np.transpose(T[i,2:dimension+2])
            b4 = np.transpose(T[i,0:dimension])
            b5 = np.transpose(T[i,1:dimension+1])
            b = 0.25*(b1+b2+b3+b4) - b5
            for j in range(dimension): # the "i" index of A should be j and the "j" index should be j,j+1 and j-1
                if j == 0:
                    A[j,j] = 1.25 
                    A[j,j+1] = -0.25
                elif j == dimension-1:
                    A[j,j] = 1.25
                    A[j,j-1] = -0.25
                else:
                    A[j,j] = 1
                    A[j,j+1] = -0.25
                    A[j,j-1] = -0.25
                    
            X = GaussianElimination(A,b)
            T[i,1:dimension+1] = T[i,1:dimension+1] + w*np.transpose(X)
            T[i,dimension+1] = 2*upperbondaryT(i) - T[i,dimension] #updating upper row of ghost cells
            T[i,0] = - T[i,1]  #updating lower row of ghost cells

        # updating left and right columns of ghost cells
        for j in range(1,dimension+1):
            T[0,j] = - T[1,j]
            T[dimension+1,j] = - T[dimension,j]

        # calculating L2
        residualsum = 0
        for i in range(1,dimension+1):
            for j in range(1,dimension+1):
                residual = abs(T[i+1,j]+T[i-1,j]+T[i,j+1]+T[i,j-1]-4*T[i,j])
                residualsum += residual**2
        L2 = (residualsum / (dimension**2))**0.5
        iteration_time = time.time()
        L2list.append(L2)
        iterationlist.append(iteration)
        timelist.append(iteration_time - start_time)
    convergence_iteration = iteration
    convergence_time = iteration_time - start_time
    #T = T[1:dimension+1,1:dimension+1]

    return T, L2list, iterationlist, timelist, convergence_iteration, convergence_time

#--------------------------------------------------------------------------------------------------------------
def implicit(dimension):


    A = np.zeros((dimension**2,dimension**2))
    d = np.zeros((dimension**2,1))
    Twall1 = 0
    Twall2 = 0
    Twall4 = 0
    def Twall3(i):
        x = (1/dimension)*(i+0.5) 
        t = np.sin(np.pi*x)
        return t

    def kp(i,j):
        return j*dimension + i

    def cornercells_coefficients(A,k,m1,m2,m3):
        A[k,m1] = 1
        A[k,m2] = 1
        A[k,m3] = -6
        return A

    def boundarycells_coefficients(A,k,m1,m2,m3,m4):
        A[k,m1] = 1
        A[k,m2] = 1
        A[k,m3] = 1
        A[k,m4] = -5
        return A

    def centralcells_coefficients(A,k,m1,m2,m3,m4,m5):
        A[k,m1] = 1
        A[k,m2] = 1
        A[k,m3] = 1
        A[k,m4] = 1
        A[k,m5] = -4
        return A

        

    start_time = time.time()
    for k in range(dimension**2):
        i = k % dimension
        j = k // dimension
        # corner coefficients---------------------------------------------------
        if k == 0: #corner left-down
            m1 = kp(i,j+1)
            m2 = kp(i+1,j)
            m3 = kp(i,j)
            A = cornercells_coefficients(A,k,m1,m2,m3)
            d[k] = -2*Twall1 -2*Twall2

        if k == dimension-1: #corner right-down
            m1 = kp(i,j+1)
            m2 = kp(i-1,j)
            m3 = kp(i,j)
            A = cornercells_coefficients(A,k,m1,m2,m3)
            d[k] = -2*Twall1 -2*Twall4

        if k == dimension*(dimension-1): #corner left-up
            m1 = kp(i,j-1)
            m2 = kp(i+1,j)
            m3 = kp(i,j)
            A = cornercells_coefficients(A,k,m1,m2,m3)
            d[k] = -2*Twall2 -2*Twall3(i)

        if k == dimension**2-1: #corner right-up
            m1 = kp(i,j-1)
            m2 = kp(i-1,j)
            m3 = kp(i,j)
            A = cornercells_coefficients(A,k,m1,m2,m3)
            d[k] = -2*Twall1 -2*Twall4
        #---------------------------------------------------------------
        #Boundary cells- excluding corner cells- construction
        if  0 < k < dimension-1: #down side
            m1 = kp(i-1,j)
            m2 = kp(i+1,j)
            m3 = kp(i,j+1)
            m4 = kp(i,j)
            A = boundarycells_coefficients(A,k,m1,m2,m3,m4)
            d[k] = -2*Twall1

        if k % dimension == 0 and k != 0 and k != dimension*(dimension-1): # left side
            m1 = kp(i+1,j)
            m2 = kp(i,j-1)
            m3 = kp(i,j+1)
            m4 = kp(i,j)
            A = boundarycells_coefficients(A,k,m1,m2,m3,m4)
            d[k] = -2*Twall2

        if k % dimension == dimension-1 and k != dimension-1 and k != dimension**2-1: # right side
            m1 = kp(i-1,j)
            m2 = kp(i,j-1)
            m3 = kp(i,j+1)
            m4 = kp(i,j)
            A = boundarycells_coefficients(A,k,m1,m2,m3,m4)
            d[k] = -2*Twall4

        if dimension**2-dimension < k < dimension**2-1: #upper side
            m1 = kp(i+1,j)
            m2 = kp(i-1,j)
            m3 = kp(i,j-1)
            m4 = kp(i,j)
            A = boundarycells_coefficients(A,k,m1,m2,m3,m4)
            d[k] = -2*Twall3(i)
        #-------------------------------------------------------------------------------------------
        # Central Cells
        if 0<i<dimension-1 and 0<j<dimension-1:
            m1 = kp(i+1,j)
            m2 = kp(i-1,j)
            m3 = kp(i,j-1)
            m4 = kp(i,j+1)
            m5 = kp(i,j)
            A = centralcells_coefficients(A,k,m1,m2,m3,m4,m5)
            d[k] = 0
        #-------------------------------------------------------------------------------------------
    X = GaussianElimination(A,d)
    T = np.zeros((dimension,dimension))
    for k in range(dimension**2):
        i = k // dimension
        j = k % dimension
        T[i,j] = X[k]
    T = np.transpose(T)
    end_time = time.time()
    CPUtime = end_time - start_time

    return T, CPUtime
#--------------------------------------------------------------------------------------------------------------
def convergence(dimension,solver,w):
    convergenceLimit = 10**(-12)
    initial_T = initialT(dimension)
    if solver.lower() == 'gs':
        result = GS(initial_T,dimension,convergenceLimit,False) 
    elif solver.lower() == 'pj':
        result = PJacobi(initial_T,dimension,convergenceLimit) 
    elif solver.lower() == 'sor':
        result = SOR(initial_T,dimension,convergenceLimit,w)
    elif solver.lower() == 'line sor':
        result = LineSOR(dimension,w)

    L2list = result[1]
    iterationlist = result[2]
    timelist = result[3]
    convergence_iteration = result[4]
    CPUtime = result[5]
    #post-proccessing----------------------------------------------------------------------
    plt.figure(1)
    plt.plot(iterationlist, L2list, '--', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('L2-Norm of Residual')
    plt.figure(2)
    plt.plot(timelist, L2list, '--', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('L2-Norm of Residual')

    return plt.figure(1), plt.figure(2), convergence_iteration,  CPUtime
#--------------------------------------------------------------------------------------------------------------
def number4_InitialVerification(dimension,solver,w):
    initial_T = initialT(dimension)
    convergenceLimit = 10**(-10)
    if solver.lower() == 'gs':
        result = GS(initial_T,dimension,convergenceLimit,False)
    elif solver.lower() == 'line sor':
        result = LineSOR(dimension,w)
    elif solver.lower() == 'implicit':
        result = implicit(dimension)
    finalT = result[0]
    if solver.lower() != 'implicit':
        finalT = finalT[1:dimension+1,1:dimension+1]
    x_values = np.linspace(1/(2*dimension),1 - 1/(2*dimension),dimension)
    y_values = np.linspace(1/(2*dimension),1 - 1/(2*dimension),dimension)
    X , Y = np.meshgrid(x_values, y_values)
    finalT = np.transpose(finalT)
    #constructing Exact solution-------------------------------------------
    def Texact(x, y):
        Tvalue = ( np.sin(math.pi*x) * np.sinh(math.pi*y) ) / np.sinh(math.pi)
        return Tvalue
    Z = Texact(X, Y)
    #Post-proccessing------------------------------------------------------
    plt.figure(1)
    ax = plt.axes(projection= '3d')
    ax.plot_surface(X, Y, finalT)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Numerical solution')
    plt.figure(2)   
    ax = plt.axes(projection= '3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Exact solution')
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------
def number5_ConvergenceHistory():
    convergence_iteration_list = []
    CPUtimelist = []
    for dimension in meshlist:
        #convergence_plotter(dimension,'gs',0)
        result = convergence(dimension,'gs',0)
        convergence_iteration_list.append(result[2])
        CPUtimelist.append(result[3])

    plt.figure(1).legend(['M1', 'M2', 'M3', 'M4'])
    plt.grid(True)
    plt.figure(2).legend(['M1', 'M2', 'M3', 'M4'])
    plt.grid(True)
    plt.show()
    print( f'convergence history for mesh list: {convergence_iteration_list}  CPU time for mesh list: {CPUtimelist}')

#--------------------------------------------------------------------------------------------------------------
def number6_OrderofAccuracy(Manipulation):
    EL1list = []
    EL2list = []
    ELinflist = []
    X = []
    for dimension in [10, 20, 30, 40]:
        initial_T = initialT(dimension)
        result1 = GS(initial_T,dimension,10**(-12),Manipulation)
        finalT = result1[0]
        result2 = ErrorNorms(finalT,dimension)
        EL1list.append(result2[0])
        EL2list.append(result2[1])
        ELinflist.append(result2[2])
        X.append(dimension**2)
    #Post-Proccessing-----------------------------------------------
    plt.xscale('log')
    plt.yscale('log')
    Xlog = np.log(X)
    L1log = np.log(EL1list)
    L2log = np.log(EL2list)
    Linflog = np.log(ELinflist)
    plt.plot(X, EL1list, marker = 'o')
    a1 = slope, intercept = np.polyfit(Xlog,L1log,1)
    plt.plot(X, EL2list, marker = 'o')
    a2 = slope, intercept = np.polyfit(Xlog,L2log,1)
    if Manipulation == False:
        plt.plot(X, ELinflist, marker = 'o')
        a3 = slope, intercept = np.polyfit(Xlog,Linflog,1)
    #plt.xticks(X)
    plt.grid(True, which = 'minor')
    plt.xlabel('#CVs')
    plt.ylabel('Error Norm')
    if Manipulation == False:
        plt.legend([f'L1, m = {a1[0]}', f'L2, m = {a2[0]}', f'Linf, m = {a3[0]}'])
    else:
        plt.legend([f'L1, m = {a1[0]}', f'L2, m = {a2[0]}'])
    plt.show()

#-------------------------------------------------------------------------------------------------------------
def number7():
    convergence_iteration_list = []
    CPUtimelist = []
    dimension = M4dimension
    #plotting convergence of Guass-Seidel--------------------------------------
    GSresult = convergence(dimension,'gs',0)
    convergence_iteration_list.append(GSresult[2])
    CPUtimelist.append(GSresult[3])
    #plotting convergence of Point Jacobi--------------------------------------
    PJresult = convergence(dimension,'pj',0)
    convergence_iteration_list.append(PJresult[2])
    CPUtimelist.append(PJresult[3])
    #plotting convergence of SOR-----------------------------------------------
    for w in [1.1, 1.3, 1.5]:
        SORresult = convergence(dimension,'sor',w)
        convergence_iteration_list.append(SORresult[2])
        CPUtimelist.append(SORresult[3])
    #Legend addition-----------------------------------------------------------
    plt.figure(1).legend(['GS','PJ','SOR-w=1.1', 'SOR-w=1.3', 'SOR-w=1.5'])
    plt.grid(True)
    plt.figure(2).legend(['GS','PJ','SOR-w=1.1', 'SOR-w=1.3', 'SOR-w=1.5'])
    plt.grid(True)
    #--------------------------------------------------------------------------
    plt.show()
    print( f'convergence history accordingly: {convergence_iteration_list}  CPU time accordingly: {CPUtimelist}')


#-----------------------------------------------------------------------------------------------------------------
def number8_1():
    w = 1.1
    convergence_iteration_list = []
    CPUtimelist = []
    #
    for dimension in meshlist:
        result = convergence(dimension,'sor',w)
        convergence_iteration_list.append(result[2])
        CPUtimelist.append(result[3])
    #
    for dimension in meshlist:
        result1 = convergence(dimension,'line sor',w)
        convergence_iteration_list.append(result1[2])
        CPUtimelist.append(result1[3])
    #Legend addition-----------------------------------------------------------
    plt.figure(1).legend(['SOR-w=1.1-M1','SOR-w=1.1-M2','SOR-w=1.1-M3', 'SOR-w=1.1-M4', 'LSOR-w=1.1-M1','LSOR-w=1.1-M2','LSOR-w=1.1-M3', 'LSOR-w=1.1-M4'])
    plt.grid(True)
    plt.figure(2).legend(['SOR-w=1.1-M1','SOR-w=1.1-M2','SOR-w=1.1-M3', 'SOR-w=1.1-M4', 'LSOR-w=1.1-M1','LSOR-w=1.1-M2','LSOR-w=1.1-M3', 'LSOR-w=1.1-M4'])
    plt.grid(True)
    #--------------------------------------------------------------------------
    plt.show()
    print( f'convergence history accordingly: {convergence_iteration_list}  CPU time accordingly: {CPUtimelist}')

#-----------------------------------------------------------------------------------------------------------------
def number8_2():
    for w in [1.3, 1.5]:
        cputimelist2 = []
        for dimension in [10,20,30]:
            T = initialT(dimension)
            result2 = SOR(T,dimension,10**(-12),w)
            cputimelist2.append(result2[5]) 
        print(f'Here are CPU time for SOR with w = {w} of mesh 1 2 3: {cputimelist2}')
    for w in [1.3, 1.5]:
        cputimelist3 = []
        for dimension in [10,20,30,40]:
            T = initialT(dimension)
            result3 = LineSOR(dimension,w)
            cputimelist3.append(result3[5]) 
        print(f'Here are CPU time for Line SOR with w = {w} of mesh 1 2 3 4: {cputimelist3}')
    cputimelist4 = []    
    for dimension in [10,20,30,40]:
        T = initialT(dimension)
        result4 = implicit(dimension)
        cputimelist4.append(result4[1])
    print(f'Here are CPU time for Implicit method of mesh 1 2 3 4: {cputimelist4}')    

#Execution-------------------------------------------------------------------------------------------------------------
if command == '1':
    number4_InitialVerification(M1dimension,'gs',0)
elif command == '2':
    number5_ConvergenceHistory()
elif command == '3':
    number6_OrderofAccuracy(False)
elif command == '4':
    number6_OrderofAccuracy(True)
elif command == '5':
    number7()
elif command == '6':
    number4_InitialVerification(M4dimension,'line sor',1.1)
elif command == '7':
    number8_1()
elif command == '8':
    number4_InitialVerification(M3dimension,'implicit',0)
elif command == '9':
    number8_2()
