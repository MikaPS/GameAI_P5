import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

pf_width = 4
pf_height = 4

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

prefabs = {
    "air": {
        "map": [
            ['-', '-', '-', '-'],
            ['-', '-', '-', '-'],
            ['-', '-', '-', '-'],
            ['-', '-', '-', '-']
        ],
        "prefered": {
            "up": [
                "air",
                "stairs",
                "ground"
            ],
            "down": [
                "pipe",
                "enemy",
                "jump_gap",
                "stairs",
                "ground"
            ],
            "sides": [
                "air",
                "jump_gap",
                "pipe",
                "enemy",
                "ground"
            ]
        }   
    },
    "jump_gap": {
        "map": [
            ['-', 'o', 'o', '-'],
            ['o', '-', '-', 'o'],
            ['-', '-', '-', '-'],
            ['X', '-', '-', 'X']
        ],
        "prefered": {
            "up": [
                "air"
            ],
            "down": [
                "enemy"
            ],
            "sides": [
                "pipe",
                "enemy",
                "stairs",
                "power_up"
            ]
        }            
    },
    "pipe": {
        "map": [
            ['-', 'T', '-', '-'],
            ['-', '|', 'o', 'T'],
            ['-', '|', '-', '|'],
            ['X', 'X', 'X', 'X']
        ],
        "prefered": {
            "up": [
                "air"
            ],
            "down": [
                "enemy",
                "jump_gap"
            ],
            "sides": [
                "enemy",
                "power_up"
            ]
        }
    },
    "enemy": {
        "map": [
            ['-', '-', '-', '-'],
            ['-', '-', 'E', '-'],
            ['-', '-', '-', '-'],
            ['X', 'X', 'X', 'X']
        ],
        "prefered": {
            "up": [
                "air",
                "jump_gap"
            ],
            "down": [
                "jump_gap"
            ],
            "sides": [
                "jump_gap",
                "pipe",
                "stairs"
            ]
        }
    },
    "power_up": {
        "map": [
            ['-', 'E', '-', '-'],
            ['B', '?', 'M', '?'],
            ['-', '-', '-', '-'],
            ['-', '-', '-', '-']
        ],
        "prefered": {
            "up": [
                "air"
            ],
            "down": [
                "enemy",
                "stairs",
                "ground",
                "pipe"
            ],
            "sides": [
                "jump_gap",
                "pipe",
                "enemy",
                "stairs"
            ]
        }
    },
    "stairs": {
        "map": [
            ['-', '-', '-', 'X'],
            ['-', '-', 'X', 'X'],
            ['-', 'X', 'X', 'X'],
            ['X', 'X', 'X', 'X']
        ],
        "prefered": {
            "up": [
                "air",
                "jump_gap"
            ],
            "down": [
                "air"
            ],
            "sides": [
                "jump_gap",
                "pipe",
                "enemy",
                "power_up"
            ]
        }
    }
    ,
    "ground": {
        "map": [
            ['-', '-', 'B', '-'],
            ['-', '-', '-', '-'],
            ['-', '-', '-', '-'],
            ['X', 'X', 'X', 'X']
        ],
        "prefered": {
            "up": [
                "air",
                "power_up",
                "jump_gap",
            ],
            "down": [
                "jump_gap",
                "ground",
                "power_up"
            ],
            "sides": [
                "ground",
                "pipe",
                "enemy",
                "stairs",
                "power_up"
            ]
        }
    }
}

def set_section(genome, x_start, y_start, prefab_name):
    for x in range(pf_width):
        for y in range(pf_height):
            genome[y_start + y][x_start + x] = prefabs[prefab_name]["map"][y][x]
    return genome

def get_random_preference(prefab_name, direction) -> str:
    return random.choice(prefabs[prefab_name]["prefered"][direction])

# foreach neighbor:
#     is it's prefab in this section's prefered list?
#     if not, replace neighbor with section from other genome

def get_section(x, y) -> tuple[int, int, int, int]:
    x1 = (x // pf_width) * pf_width
    y1 = (y // pf_height) * pf_height
    return (x1, y1, x1 + pf_width, y1 + pf_height)


def get_prefab(genome, section: tuple[int, int, int, int])-> str | None:
    for prefab_name in prefabs:
        matches = True
        prefab_map = prefabs[prefab_name]["map"]
        for x in range(pf_width):
            for y in range(pf_height):
                if prefab_map[y][x] != genome[y + section[1]][x + section[0]]:
                    matches = False
        if matches:
            return prefab_name
    return None

def get_neighbors(section: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    neighbors = {}
    if section[0] > 0: # left
        neighbors["left"] = (section[0] - pf_width, section[1], section[2] - pf_width, section[3])
    if section[2] < width: # right
        neighbors["right"] = (section[0] + pf_width, section[1], section[2] + pf_width, section[3])
    if section[3] < height: # down
        neighbors["down"] = (section[0], section[1] + pf_height, section[2], section[3] + pf_height)
    if section[1] > 0: # up
        neighbors["up"] = (section[0], section[1] - pf_height, section[2], section[3] - pf_height)

    return neighbors

# The level as a grid of tiles
class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome: list[list[str]]):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # TODO: Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self) -> float | int | None:
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome: list[list[str]]) -> list[list[str]]:
        # TODO: implement a mutation operator, also consider not mutating this individual
        # TODO: also consider weighting the different tile types so it's not uniformly random
        # TODO: consider putting more constraints on this to prevent pipes in the air, etc
        
        # genome is most likely a list/matrix, can be accessed with []
        left = 1   # left most column of level
        right = width - 1 # right-most column of level
        
        visited = {}

        for y in range(height): 
            for x in range(left, right):
                if random.random() < 0.2:
                
                    section = get_section(x, y)
                    prefab_name = get_prefab(genome, section)
                    if prefab_name == None:
                        rand_prefab = random.choice(list(prefabs.keys()))
                        genome = set_section(genome, section[0], section[1], rand_prefab)
                        continue
                    
                    if section not in visited:
                        neighbors = get_neighbors(section)

                        for direction in neighbors:
                            neighbor = neighbors[direction]
                            direction = "sides" if direction == "left" or direction == "right" else direction
                            if get_prefab(genome, neighbor) not in prefabs[prefab_name]["prefered"][direction]:
                                genome = set_section(genome, neighbor[0], neighbor[1], get_random_preference(prefab_name, direction))
                        
                        visited[section] = True

            if y < 4:
                section = get_section(x, y)
                prefab_name = get_prefab(genome, section)
                if prefab_name == "stairs" or prefab_name == "pipe":
                    genome = set_section(genome, neighbor[0], neighbor[1], get_random_preference(prefab_name, direction))

            
        return genome

    # Create zero or more children from self and other
    def generate_children(self, other): # other: Individual_Grid
        c1_genome = copy.deepcopy(self.genome)
        c2_genome = copy.deepcopy(self.genome)
        # Leaving first and last columns alone...
        # do crossover with other
        left = 1
        right = width - 1
        visited = {}
        # print("genome: ", new_genome)
        for y in range(height):
            for x in range(left, right):
                # TODO: Which one should you take?  Self, or other?  Why?
                # TODO: consider putting more constraints on this to prevent pipes in the air, etc
                cur_genome = c1_genome if random.random() > 0.5 else c2_genome
                # SECTION PREFAB
                section = get_section(x, y)
                prefab_name = get_prefab(cur_genome, section)
                if prefab_name == None:
                    rand_prefab = random.choice(list(prefabs.keys()))
                    cur_genome = set_section(cur_genome, section[0], section[1], rand_prefab)
                    continue
                
                if section not in visited:
                    neighbors = get_neighbors(section)

                    for direction in neighbors:
                        neighbor = neighbors[direction]
                        direction = "sides" if direction == "left" or direction == "right" else direction
                        if get_prefab(cur_genome, neighbor) not in prefabs[prefab_name]["prefered"][direction]:
                            other_prefab = get_prefab(other.genome, neighbor)
                            if other_prefab == None:
                                cur_genome = set_section(cur_genome, neighbor[0], neighbor[1], get_random_preference(prefab_name, direction))
                                continue
                            cur_genome = set_section(cur_genome, neighbor[0], neighbor[1], other_prefab)
                    
                    visited[section] = True
                
        # do mutation; note we're returning a one-element tuple here
        c1_genome = self.mutate(c1_genome)
        c1_genome[14][0] = "m"
        c1_genome[15][0] = "X"
        c1_genome[15][1] = "X"
        c1_genome[7][-1] = "v"
        for col in range(8, 14):
            c1_genome[col][-1] = "f"
        for col in range(14, 16):
            c1_genome[col][-1] = "X"

        c2_genome = self.mutate(c2_genome)
        c2_genome[14][0] = "m"
        c2_genome[15][0] = "X"
        c2_genome[15][1] = "X"
        c2_genome[7][-1] = "v"
        for col in range(8, 14):
            c2_genome[col][-1] = "f"
        for col in range(14, 16):
            c2_genome[col][-1] = "X"
        return (Individual_Grid(c1_genome),Individual_Grid(c2_genome))

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # TODO: Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # TODO: consider putting more constraints on this to prevent pipes in the air, etc
        # TODO: also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        for x in range(0, width, pf_width):
            for y in range(0, height, pf_height):
                g = set_section(g, x, y, random.choice(list(prefabs.keys())))
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    '''
        sub or add to val, using a bell curve distribution to pick 
        values, using variance as the SD. min and max to clamp value.
    '''
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf

class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # TODO: Add more metrics?
        # TODO: Improve this with any code you like
        coefficients = dict(
            meaningfulJumpVariance=0.8,
            negativeSpace=0.1,
            pathPercentage=0.6,
            emptyPercentage=0.1,
            linearity=0.5,
            solvability=5.0
        )
        penalties = 0
        # TODO: For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 6:
            penalties -= 2
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) <= 3:
            penalties -= 1
        if len(list(filter(lambda de: de[1] == "1_platform", self.genome))) < 10:
            penalties -= 2
        if len(list(filter(lambda de: de[1] == "0_hole" and de[2] == 3, self.genome))) > 3:
            penalties -= 1
        # TODO: If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # TODO: How does this work?  Explain it in your writeup.
        # TODO: consider putting more constraints on this, to prevent generating weird things

        # genome is made from [(int, string, int, bool)]
                            # [(x, type of block, y value, has power up)]
        # only mutate 10% of the time?
        if random.random() < 0.2 and len(new_genome) > 0:
            to_change = random.randint(0, len(new_genome) - 1) # pick a random block to mutate
            de = new_genome[to_change]
            new_de = de
            x = de[0]
            de_type = de[1]
            choice = random.random()
            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    breakable = not de[3]
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                y = de[2]
                has_powerup = de[3]  # boolean
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    has_powerup = not de[3]
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2) 
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4) 
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]  # -1 or 1
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=height - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, 1)
            elif de_type == "1_platform":
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                else:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
            new_genome.pop(to_change)
            heapq.heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):
        # TODO: How does this work?  Explain it in your writeup.
        # pick random slice from each parent and combine them to create 2 children
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T" # where the pipe starts?
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # TODO: Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # TODO: Maybe enhance this
        elt_count = random.randint(20, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 3)), # x_start, width
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])), # x_start, width, height, block type
            (random.randint(1, width - 2), "2_enemy"), # x_coord
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)), # x, y
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])), # x, y, is_breakable
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])), # x, y, has_mushroom
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), 1), # x, height, direction (ascending/descending)
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 13)) # x, height
        ]) for i in range(elt_count)]
        return Individual_DE(g)

#############################################################################

Individual = Individual_Grid
# Individual = Individual_DE

# TODO
def generate_successors(population: list[Individual_Grid]) -> list[Individual_Grid]:
    results = []
    # TODO: Design and implement this
    # Hint: Call generate_children() on some individuals and fill up results.
    # print("population: ", population)
    population = sorted(population, key=lambda ind: ind.fitness(), reverse=True)
    
    selected: list[Individual_Grid] = []
    for individual in population:
        if random.random() < 0.5:
            selected.append(individual)
    # generate children returns a one-element tuple with the individual grid with the new genome
        # which seems to be the singleton grid
    for i in range(0, len(selected), 2):
        if i + 1 >= len(selected):
            break
        results += list(selected[i].generate_children(selected[-1-i]))
        
    return results


def ga():
    # TODO: Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # TODO: (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.1
                      else Individual.random_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                best = Individual.empty_individual()
                if len(population) <= 3:
                    break
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1

                # TODO: Determine stopping condition
                stop_condition = False # best.fitness() > 2
                if stop_condition:
                    break
                # TODO: Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # TODO: You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
