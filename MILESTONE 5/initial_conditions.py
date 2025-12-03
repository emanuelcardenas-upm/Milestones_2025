from numpy import linspace, zeros, reshape

def initial_conditions(Nc, Nb): 
 
    U0 = zeros(2*Nc*Nb)
    U1 = reshape(U0, (Nb, Nc, 2))  
    r0 = reshape(U1[:, :, 0], (Nb, Nc))     
    v0 = reshape(U1[:, :, 1], (Nb, Nc))
     
    r0[0,:] = [1, 0, 0]
    v0[0,:] = [0, 1, 0]
         
    r0[1,:] = [-1, 0, 0]
    v0[1,:] = [0, -1, 0] 
     
    r0[2,:] = [0, 1, 0] 
    v0[2,:] = [-1, 0, 0]       
    
    return U0