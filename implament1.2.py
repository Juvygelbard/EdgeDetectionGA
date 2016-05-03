import numpy as np
import multiprocessing as mp

SQ_SIZE = 8
PIC_PATH = 'test64.bmp'
PIC_SIZE = 64
NUM_GENERATIONS = 200
POP_SIZE = 50

class DissimilarityRegion():
    def __init__(self, struct, dirs):
        self.__convert_to_pos__(np.array(struct))
        self.__convert_dirs__(dirs)

    def __convert_to_pos__(self, struct):
        # find c pos
        c_pos = zip(*np.where(struct=='c')).__next__()
        # build poses
        r1 = [(x-c_pos[1], y-c_pos[0]) for (y, x) in zip(*np.where(struct=='1'))]
        r2 = [(x-c_pos[1], y-c_pos[0]) for (y, x) in zip(*np.where(struct=='2'))]
        Xs = [(x-c_pos[1], y-c_pos[0]) for (y, x) in zip(*np.where(struct=='x'))]
        self.__r1__ = np.array(r1)
        self.__r2__ = np.array(r2)
        self.__Xs__ = np.array(Xs)

    def __convert_dirs__(self, dirs):
        conv_map = {'U': (-1, 0),
                    'D': (1, 0),
                    'L': (0, -1),
                    'R': (0, 1),
                    'UL': (-1, -1),
                    'UR': (-1, 1),
                    'DL': (1, -1),
                    'DR': (1, 1)}
        self.__dirs__ = [conv_map[x] for x in dirs]

    def calculate(self, mat, x_pos, y_pos, alpha=1):
        r1_vals = [mat[y_pos+y][x_pos+x] for x, y in self.__r1__ if 0 <= y_pos+y < len(mat) and 0 <= x_pos+x < len(mat[0])]
        r2_vals = [mat[y_pos+y][x_pos+x] for x, y in self.__r2__ if 0 <= y_pos+y < len(mat) and 0 <= x_pos+x < len(mat[0])]
        r1, r2 = 255, 255
        if(len(r1_vals)>0):
            r1 = sum(r1_vals)/len(r1_vals)
        if(len(r2_vals)>0):
            r2 = sum(r2_vals)/len(r2_vals)
        return abs(r2-r1) * alpha

    def calculate_shifted(self, mat, x_pos, y_pos, alpha=1):
        new_pos = [(x_pos+x, y_pos+y) for (y, x) in self.__dirs__]
        new_val = [self.calculate(mat, x, y, alpha) for (x, y) in new_pos]
        return max(new_val)

# building basis set
if __name__ == '__main__':
    bss1 = [[1, 'x', 2, 2],
            [1, 'c', 2, 2],
            [1, 'x', 2, 2]]

    dr1 = DissimilarityRegion(bss1, ['L', 'R'])

    bss2 = [[2, 2, 2],
            [2, 2, 2],
            ['x', 'c', 'x'],
            [1, 1, 1]]

    dr2 = DissimilarityRegion(bss2, ['U', 'D'])

    bss3 = [[0, 1, 'x', 0],
            [1, 'c', 2, 2],
            ['x', 2, 2, 0],
            [0, 2, 0, 0]]

    dr3 = DissimilarityRegion(bss3, ['UL', 'DR'])

    bss4 = [[0, 2, 0, 0],
            ['x', 2, 2, 0],
            [1, 'c', 2, 2],
            [0, 1, 'x', 0]]

    dr4 = DissimilarityRegion(bss4, ['UR', 'DL'])

    bss5 = [[0, 2, 0],
            ['x', 1, 2],
            [1, 'c', 2],
            [1, 'x', 2]]

    dr5 = DissimilarityRegion(bss5, ['U', 'D', 'L', 'R'])

    bss6 = [[0, 2, 2],
            ['x', 2, 2],
            [1, 'c', 'x']]

    dr6 = DissimilarityRegion(bss6, ['U', 'D', 'L', 'R'])

    bss7 = [['x', 2, 2],
            ['c', 2, 2],
            [1, 'x', 0]]

    dr7 = DissimilarityRegion(bss7, ['U', 'D', 'L', 'R'])

    bss8 = [[1, 'x', 2],
            [1, 'c', 2],
            ['x', 2, 2]]

    dr8 = DissimilarityRegion(bss8, ['U', 'D', 'L', 'R'])

    bss9 = [[0, 1, 0],
            [1, 'x', 2],
            ['c', 2, 2],
            ['x', 2, 2]]

    dr9 = DissimilarityRegion(bss9, ['U', 'D', 'L', 'R'])

    bss10 = [[0, 2, 2, 0],
            [2, 2, 'x', 1],
            ['x', 'c', 1, 0]]

    dr10 = DissimilarityRegion(bss10, ['U', 'D', 'L', 'R'])

    bss11 = [[2, 2, 2],
            [2, 'c', 'x'],
            ['x', 1, 1]]

    dr11 = DissimilarityRegion(bss11, ['U', 'D', 'L', 'R'])

    bss12 = [[2, 2, 2, 0],
            ['x', 'c', 1, 2],
            [1, 1, 'x', 0]]

    dr12 = DissimilarityRegion(bss12, ['U', 'D', 'L', 'R'])

    bss = [dr1, dr2, dr3, dr4, dr5, dr6, dr7, dr8, dr9, dr10, dr11, dr12]

def compute_dissimilarity(image, bss):
    diss_map = np.zeros((len(image), len(image[0])))
    for y in range(len(image)):
        for x in range(len(image[y])):
            diss = [dr.calculate(image, x, y) for dr in bss]
            max_diss = max(diss)
            max_pos = [pos for pos, x in enumerate(diss) if x==max_diss][0]
            max_shifted = bss[max_pos].calculate_shifted(image, x, y)
            if(max_diss >= max_shifted):
                delta = max_diss/3
                diss_map[y, x] += delta
                for (dx, dy) in bss[max_pos].__Xs__:
                    if 0 <= y+dy < len(diss_map) and 0 <= x+dx < len(diss_map[0]):
                        diss_map[y+dy][x+dx] += delta
    # normalize
    max_val = diss_map.max()
    for y in range(len(diss_map)):
        for x in range(len(diss_map[y])):
            diss_map[y][x] /= max_val
    return diss_map


from scipy.misc import imread
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    img = imread(PIC_PATH, mode='I')

    # compute dissimilarity map
    print("Computing Dissimilarity Map...")
    diss_map = compute_dissimilarity(img, bss)

    p_file = open('diss_map.pickle', 'wb')
    pickle.dump(diss_map, p_file)
    p_file.close()
    
    # show images
    plt.imshow(img, cmap='Greys_r')
    plt.show()
    plt.imshow(diss_map, cmap='Greys_r')
    plt.show()
else:
    p_file = open('diss_map.pickle', 'rb')
    diss_map = pickle.load(p_file)
    p_file.close()

# cost factors functions

def extrect_env(e_mat, x, y):
    env_d = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    env = []
    for (dx, dy) in env_d:
        if 0 <= y+dy < len(e_mat) and 0 <= x+dx < len(e_mat[0]):
            env.append(e_mat[y+dy][x+dx])
        else:
            env.append(0)
    return env

def curvature_factor(e_mat, x, y):
    env = extrect_env(e_mat, x, y)
    if(sum(env) <= 1):  # return 0 if it's an endpoint
        return 0
    env += [env[0], env[1]]
    for i in range(1, len(env)-1): # look for 45 deg
        if(env[i-1] + env[i] == 2):
            return 1
    for i in range(2, len(env)): # look for 90 deg
        if(env[i-2] + env[i] == 2):
            return 0.5
    return 0 # for 135 or 0, return 0

def dissimilarity_factor(e_mat, x, y, diss_map=diss_map):
    if(e_mat[y][x] == 1):
        return 0 # not an edge pixel
    else:
        return diss_map[y][x] # is an edge pixel

def num_of_edge_factor(e_mat, x, y):
    return e_mat[y][x]  # penalize for every edge pixel

def fragmentation_factor(e_mat, x, y):
    env = extrect_env(e_mat, x, y)
    surrounding_edges = sum(env)
    if(surrounding_edges == 0): # isolated endpoint
        return 1
    elif (surrounding_edges == 1): # nonisolated endpoint
        return 0.5
    else: # not an endpoint
        return 0

def thickness_factor(e_mat, x, y):
    env = extrect_env(e_mat, x, y)
    env += [env[0], env[1]]
    for i in range(1, len(env)-1): # look for 1st kind of len 3 cycles
        if(env[i-1] + env[i] == 2):
            return 1
    for i in range(1, 5): # look for 2nd kind of len 3 cycles
        if(env[(i-1)*2+1] + env[i*2+1] == 2):
            return 1
    return 0 # thin edge

factors = [(0.5, curvature_factor),
           (2, dissimilarity_factor),
           (1, num_of_edge_factor),
           (3, fragmentation_factor),
           (6.51, thickness_factor)]

def calc_cost(e_mat, factors, x1=0, y1=0, x2=0, y2=0):
    if(x2==0 and y2==0):
        y2 = len(e_mat)
        x2 = len(e_mat[0])
    f_sum = 0
    for y in range(y1, y2):
        for x in range(x1, x2):
            for (w, f) in factors:
                f_sum += w*f(e_mat, x, y)
    return f_sum

def generate_random_population(height, width, n=50):
    ans = []
    for i in range(n):
        ans.append(np.random.random_integers(0, 1, (height, width)))
    return ans

def rank_selection(size):
    total = sum(range(size+1))
    prob = [(1+i)/total for i in range(size)]
    prob.reverse()
    parents = np.random.choice(size, 2, replace=False, p=prob)
    return parents[0], parents[1]

def crossover(mat1, mat2):
    y1 = np.random.randint(0, len(mat1)-1)
    x1 = np.random.randint(0, len(mat1[0])-1)
    y2 = np.random.randint(y1+1, len(mat1))
    x2 = np.random.randint(x1+1, len(mat1[0]))
    ans1 = np.copy(mat1)
    ans2 = np.copy(mat2)
    for i in range(5):
        ans1[y1:y2, x1:x2] = mat2[y1:y2, x1:x2]
        ans2[y1:y2, x1:x2] = mat1[y1:y2, x1:x2]
        if(not np.array_equal(mat1, ans1)): # check that actual exchanges were made
            return ans1, ans2
    raise ValueError


def mutate(e_mat, p=0.08):
    height = len(e_mat)
    width = len(e_mat[0])
    new_mat = np.copy(e_mat)
    for y in range(height):
        for x in range(width):
            if(np.random.random() < p):
                if(new_mat[y][x]==0):
                    new_mat[y][x] = 1
                else:
                    new_mat[y][x] = 0
    return new_mat

def plant_in(c_from, c_to, x1, y1, x2, y2):
    c_to[y1:y2, x1:x2]=c_from[0:y2-y1, 0:x2-x1]
    return c_to

# expecting population to be sorted by cost!
def area_worker(best_map, population, x1, y1, revert_prob):
    y2 = len(population[0]) + y1
    x2 = len(population[0][0]) + x1
    pop_size = len(population)
    # sort parents by cost
    area_cost = lambda area: calc_cost(plant_in(area, best_map, x1, y1, x2, y2),
                                       factors, x1, y1, x2, y2)
    # creatng next generation
    new_population = []
    # crossovers:
    for i in range(round(pop_size/2)):
        offspring1, offspring2 = None, None
        while(offspring1 is None):
            try:
                parent1, parent2 = rank_selection(pop_size)
                offspring1, offspring2 = crossover(population[parent1], population[parent2])
            except ValueError:
                pass
        new_population += [offspring1, offspring2]
    # mutations:
    for i in range(pop_size):
        new_mat = mutate(new_population[i])
        before_cost = area_cost(population[i])
        after_cost = area_cost(new_mat)
        if(after_cost < before_cost): # check if mutated chromozome is better
            new_population[i] = new_mat
        elif(np.random.random() > revert_prob): # if it's not better, there is a [reveret_prob] chance of reverting
            new_population[i] = new_mat
    new_population = sorted(new_population, key=area_cost)
    return new_population

def simulate(population, generations, revert_prob=0.85):
    pop_size = len(population)
    height = len(population[0])
    width = len(population[0][0])
    b_height = round(height/SQ_SIZE)
    b_width = round(width/SQ_SIZE)
    population = sorted(population, key=lambda x: calc_cost(x, factors))
    for gen in range(generations):
        print("Generation: %i" %(gen+1))
        print("Best Fitness: %f" %(calc_cost(population[0], factors)))
        if __name__ == '__main__':
            pool = mp.Pool()
            new_pop_raw = []
            # break down the current maps
            for y in range(b_height):
                new_pop_raw.append([])
                for x in range(b_width):
                    area_pop = [population[i][y*SQ_SIZE:(y+1)*SQ_SIZE, x*SQ_SIZE:(x+1)*SQ_SIZE] for i in range(pop_size)]
                    new_pop_raw[y].append(pool.apply_async(area_worker, [population[0], area_pop, x*SQ_SIZE, y*SQ_SIZE, revert_prob]))
            # bring them back together
            pool.close()
            pool.join()
            new_pop = []
            for i in range(pop_size):
                child = np.zeros((height, width))
                for y in range(b_height):
                    for x in range(b_width):
                        child[y*SQ_SIZE:(y+1)*SQ_SIZE, x*SQ_SIZE:(x+1)*SQ_SIZE] = new_pop_raw[y][x].get()[i][0:SQ_SIZE, 0:SQ_SIZE]
                new_pop.append(child)
            population = new_pop
    return population

if __name__ == '__main__':
    # generate initial population
    population = generate_random_population(PIC_SIZE, PIC_SIZE, POP_SIZE)
    
    # p_file = open('last_run.pickle', 'rb')
    # population = pickle.load(p_file)
    # p_file.close()
    
    # simulate
    ans = simulate(population, NUM_GENERATIONS)
    
    # save results to pickle for lated use
    p_file = open('last_run.pickle', 'wb')
    pickle.dump(ans, p_file)
    p_file.close()
    
    # show results
    plt.imshow(ans[0], cmap='Greys_r')
    plt.show()