import numpy as np
from numba import njit
import matplotlib.pyplot as plt





@njit(fastmath=True)
def initialize_neighbors(i, j, h=9, w=9):
    """Create a list of indexes of the neighbors for postion (i,j)
    """
    neighbors_indexes = list()
    # indexes of the neighbors along the vertical line
    for i_ in range(9):
        if i_ != i:
            index = i_*w + j 
            neighbors_indexes.append(index)

    # indexes of the neighbors along the horizontal line
    for j_ in range(9):
        if j_ != j:
            index = i*w + j_ 
            neighbors_indexes.append(index)
    
    i_center, j_center = 3*(i//3) + 1, 3*(j//3) +1
    
    # indexes of the neighbors for the coressponding cube(3*3) with center (i_center, j_center)
    for i_ in range(9):
        for j_ in range(9):
            if abs(i_center - i_) <= 1 and abs(j_center - j_) <= 1:
                index = i_*w + j_
                neighbors_indexes.append(index)
    # delete repeated indexes
    neighbors_indexes = list(set(neighbors_indexes))
    # delete index with coressponding to position (i, j)
    neighbors_indexes.remove(i*w + j)
    return neighbors_indexes

@njit(fastmath=True)
def initialize_q(sudoku, height,width,n_labels):
    """
    if sudoku[i, j]!= 0, then assign for all k(k!=sudoku[i,j]) q[i,j,k]=False
    """
    q = np.zeros((height,width,n_labels), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            if sudoku[i, j]==0:
                q[i, j, :] = 1
            else:
                q[i, j, sudoku[i,j]-1] = 1
    return q


@njit(fastmath=True)
def recalculate_g(sudoku, neighbours_array, height,width,n_labels, n_neighbours):
    g = np.zeros((height,width,n_neighbours,n_labels,n_labels),dtype=np.int32)
    # if sudoku[i, j]!= 0, then all its neighbors can take value sudoku[i, j]                   
    for i in range(height):
        for j in range(width):
            neighbours = neighbours_array[i,j, :]
            current_value = sudoku[i,j]-1
            for n, neighbor in enumerate(neighbours):
                #print(neighbor)
                i_ = int(neighbor // width)
                j_ = int(neighbor % width)
                neighbour_value = sudoku[i_,j_] - 1
                # and if neighbour cell also not empty and they are not equal
                if sudoku[i,j] != 0:
                    if sudoku[i_,j_] != 0 and current_value != neighbour_value:
                        g[i, j, n, current_value,neighbour_value] = 1
                    else:
                        # if neighbour cell is empty
                        g[i,j,n,sudoku[i,j]-1,:] = 1
                         # except same value
                        g[i,j,n,sudoku[i,j]-1,sudoku[i,j]-1] = 0
                else:
                    # if (i,j) is empty and neighbour is not
                    if sudoku[i_,j_] != 0:
                        # from empty cell to non-empty 
                        g[i,j,n,:,neighbour_value] = 1
                        # except repeated neihbour value
                        g[i,j,n,neighbour_value,neighbour_value] = 0
                    else:
                        # if (i,j) empty and neighbour also
                        g[i,j,n,:,:] = 1
    return g


@njit(fastmath=True)
def is_new_label_consistent(q, g, neighbours_array, ):
    q_updated = q.copy()
    g_updated = g.copy()
    height, width, n_neighbours, n_labels, _ = g_updated.shape
    while True:
        for i in range(height):
            for j in range(width):
                for k in range(n_labels):
                    for n, neighbor in enumerate(neighbours_array[i,j,:]):
                        i_ = int(neighbor // width)
                        j_ = int(neighbor % width)
                        # 'and' elementwise and 'or' for all elements
                        is_any = bool((g[i,j,n,k,:] * q[i_,j_,:]).any())
                        if is_any:
                            g_updated[i,j,n,k,:] = g[i, j, n, k, :]  * q_updated[i_, j_, :]
                        else:
                            q_updated[i, j, k] = 0
                            g_updated[i, j, :, k, :] = 0
                            break
                            
        
        # if without changing
        if (g_updated == g).all() and (q_updated == q).all():
            # all = False
            if (g_updated == 0).all():
                return False
            # not all = False
            else:
                return True 
        else:
            # update q, g
            q = np.copy(q_updated)
            g = np.copy(g_updated)


def get_example(i):
    n = len(examples)
    
    if 0 <= i < n:
        return examples[i]
    else:
        print(f"exmaples in [0,{n}]")
        return None
            
        
examples = []
examples.append( np.array([[ 5, 3, 0, 0, 7, 0, 0, 0, 0],     
                         [ 6, 0, 0, 1, 9, 5, 0, 0, 0],       
                         [ 0, 9, 8, 0, 0, 0, 0, 6, 0],    
                         [ 8, 0, 0, 0, 6, 0, 0, 0, 3],    
                         [ 4, 0, 0, 8, 0, 3, 0, 0, 1],   
                         [ 7, 0, 0, 0, 2, 0, 0, 0, 6],     
                         [ 0, 6, 0, 0, 0, 0, 2, 8, 0],    
                         [ 0, 0, 0, 4, 1, 9, 0, 0, 5],   
                         [ 0, 0, 0, 0, 8, 0, 0, 7, 9]])   )


examples.append( np.array([[ 6, 2, 0, 9, 5, 1, 0, 0, 0],     
                        [ 0, 0, 5, 4, 0, 0, 1, 9, 6],       
                        [ 9, 0, 1, 0, 7, 0, 0, 2, 8],    
                        [ 7, 0, 4, 2, 0, 0, 8, 0, 1],    
                        [ 0, 8, 0, 6, 1, 0, 4, 3, 0],   
                        [ 1, 0, 9, 0, 4, 3, 2, 0, 0],     
                        [ 8, 1, 0, 7, 0, 4, 0, 0, 2],    
                        [ 0, 5, 7, 0, 0, 9, 3, 8, 0],   
                        [ 0, 9, 0, 0, 8, 2, 6, 1, 0]])   )


examples.append( np.array([[2, 6, 4, 0, 0, 3, 0, 0, 1],
                        [0, 9, 5, 8, 0, 0, 3, 6, 0],
                        [1, 0, 0, 0, 5, 0, 7, 0, 0],
                        [0, 0, 0, 0, 2, 5, 6, 9, 0],
                        [0, 0, 0, 4, 0, 8, 0, 0, 0],
                        [0, 8, 2, 3, 7, 0, 0, 0, 0],
                        [0, 0, 9, 0, 3, 0, 0, 0, 4],
                        [0, 1, 6, 0, 0, 2, 9, 7, 0],
                        [3, 0, 0, 9, 0, 0, 5, 2, 6]])   )

examples.append( np.array([[ 8, 1, 0, 0, 3, 0, 0, 2, 7],     
                        [ 0, 6, 2, 0, 0, 0, 0, 9, 0],       
                        [ 0, 7, 0, 0, 0, 0, 0, 0, 0],    
                        [ 0, 0, 0, 6, 0, 0, 1, 0, 0],    
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 4],   
                        [ 0, 0, 8, 0, 0, 5, 0, 7, 0],     
                        [ 0, 0, 0, 0, 0, 0, 0, 8, 0],    
                        [ 0, 0, 0, 0, 1, 0, 7, 5, 0],   
                        [ 0, 0, 0, 0, 7, 0, 0, 4, 2]])    )

    
    

def render_sudoku(sudoku):
    
    char_h, char_w = refs[0].shape[:2]
    h,w = sudoku.shape[:2]
    
    temp = refs[ sudoku[0,0] ]
    for j in range(1, w):
        temp = np.concatenate([temp, refs[sudoku[0, j]]], axis=1)
        
        if (j+1)%3 == 0 and j+1 < w:
            h_line = np.ones((char_h,1), dtype=np.uint8)
            temp = np.concatenate([temp,h_line], axis=1)
        
    res = np.copy(temp)
    #res = np.concatenate([res, temp], axis=0)
    
    for i in range(1,h):    
        temp = refs[ sudoku[i,0] ]
        for j in range(1, w):
            temp = np.concatenate([temp, refs[sudoku[i, j]] ], axis=1)
            if (j+1)%3 == 0 and j+1 < w:
                h_line = np.ones((char_h,1), dtype=np.uint8)
                temp = np.concatenate([temp,h_line], axis=1)
                
        res = np.concatenate([res, temp], axis=0)
        if (i+1)%3 == 0 and i+1 < h:
            v_line = np.ones((1,char_w*w+2), dtype=np.uint8)
            res = np.concatenate([res,v_line], axis=0)
        
    plt.imshow(res, cmap='gray')
    return None    
            

zero = np.array([
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
])

one = np.array([
    [0,0,0,0,0],
    [0,0,1,0,0],
    [0,1,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,1,1,1,0],
    [0,0,0,0,0]
])

two = np.array([
    [0,0,0,0,0],
    [0,1,1,1,0],
    [0,0,0,1,0],
    [0,1,1,1,0],
    [0,1,0,0,0],
    [0,1,1,1,0],
    [0,0,0,0,0]
])

three = np.array([
    [0,0,0,0,0],
    [0,1,1,1,0],
    [0,0,0,1,0],
    [0,1,1,1,0],
    [0,0,0,1,0],
    [0,1,1,1,0],
    [0,0,0,0,0]
])

four = np.array([
    [0,0,0,0,0],
    [0,1,0,1,0],
    [0,1,0,1,0],
    [0,1,1,1,0],
    [0,0,0,1,0],
    [0,0,0,1,0],
    [0,0,0,0,0]
])

five = np.array([
    [0,0,0,0,0],
    [0,1,1,1,0],
    [0,1,0,0,0],
    [0,1,1,1,0],
    [0,0,0,1,0],
    [0,1,1,1,0],
    [0,0,0,0,0]
])

six = np.array([
    [0,0,0,0,0],
    [0,1,1,1,0],
    [0,1,0,0,0],
    [0,1,1,1,0],
    [0,1,0,1,0],
    [0,1,1,1,0],
    [0,0,0,0,0]
])

seven = np.array([
    [0,0,0,0,0],
    [0,1,1,1,0],
    [0,0,0,1,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,0,0,0]
])

eight = np.array([
    [0,0,0,0,0],
    [0,1,1,1,0],
    [0,1,0,1,0],
    [0,1,1,1,0],
    [0,1,0,1,0],
    [0,1,1,1,0],
    [0,0,0,0,0]
])

nine = np.array([
    [0,0,0,0,0],
    [0,1,1,1,0],
    [0,1,0,1,0],
    [0,1,1,1,0],
    [0,0,0,1,0],
    [0,1,1,1,0],
    [0,0,0,0,0]
])

refs = np.array([zero, one, two, three, four, five, six, seven, eight, nine])