import random
#add functions like species management for proper NEAT algorithm
#this is just a partial implementation for NEAT algo 
"""
Represents a node in the graph. Each node has a unique ID,
a random value within the range (-0.5, 0.5), a set of connections to other nodes, and a set of nodes that have a connection to this node.
"""
class Node:
    def __init__(self, node_id, connections=[], back=[]) -> None:
        self.id = node_id
        value_range=(-0.5, 0.5)
        self.value = random.uniform(*value_range)
        self.connections = set(connections)
        self.back_connections = set(back)


"""
Represents a graph data structure, where each node is identified by a unique ID and can have connections to other nodes.

The `graph` class provides methods to add and remove nodes, add edges between nodes, and retrieve information about the connections of a node.
"""
class graph:
    def __init__(self) -> None:
        self.nodes = {}  

    def add_node(self, node_id) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, [],[])
    
    def get_value(self,node_id) -> int:
        if node_id in self.nodes:
            return self.nodes[node_id].value
        else :
            pass
            #print("not exist")

    def add_edge(self, node_id1, node_id2) -> None:
        if node_id2 not in self.nodes[node_id1].connections and node_id1 not in self.nodes[node_id2].back_connections:
            self.nodes[node_id1].connections.add(node_id2)
            self.nodes[node_id2].back_connections.add(node_id1)
        elif node_id2 not in self.nodes[node_id1].connections and node_id1 in self.nodes[node_id2].back_connections:
                self.nodes[node_id1].connections.add(node_id2)  # can modify it to add or remove the connection randomly
            #print("already exist")
        else :
            pass
            #print("already exist")
    
    def get_connections(self, node_id) -> list:
        if node_id in self.nodes:
            return self.nodes[node_id].connections
        else :
            pass
            #print("not exist")
    
    def get_back_connections(self,node_id)->list:
        if node_id in self.nodes:
            return self.nodes[node_id].back_connections
        else :
            pass
            #print("not exist")

    def remove_node(self, node_id) -> None:
            if node_id in self.nodes:
                #add code to remove the connections from connected nodes
                self.nodes.pop(node_id)
            else :
                pass
                #print("not exist")



"""
Represents a genome in the genetic algorithm. A genome is a single individual in the population, and contains a graph representation of a neural network.

The genome class is responsible for:
- Creating the initial structure of the neural network, including input, hidden, and output layers.
- Initializing the forward connections between the layers.
- Performing a forward pass through the neural network to compute the outputs.
- Randomly adding new nodes to the hidden layer to introduce structural mutations.

The genome class is a core component of the NEAT (Neuroevolution of Augmenting Topologies) algorithm,
which evolves the structure and weights of neural networks through a genetic algorithm.
"""

class genome:
    def __init__(self, ip,op,constr=1) -> None:
        if constr==0:
            self.id = 0
            self.graph=graph()
            self.input_layer=[]
            self.fitness = 0
    
            for _ in range(ip):
                self.input_layer.append(self.create_node())
        #print(self.graph.nodes)

            self.hidden_layer=[]
            for _ in range(random.randint(2, 5)):
                self.hidden_layer.append(self.create_node())
        
            self.hidden_layer_2=[]

            self.output_layer=[]
            for _ in range(op):
                self.output_layer.append(self.create_node())
        
            self.forward_conn()
            self.add_node()
        
        #self.remove_unconnected_nodes()
        else:
            self.id = 0
            self.graph=graph()
            self.input_layer=[]
            self.fitness = 0
            self.hidden_layer=[]
            self.hidden_layer_2=[]
            self.output_layer=[]
    
    def create_node(self) -> int:
        id_i = self.id
        self.graph.add_node(id_i)
        self.id += 1
        return id_i
        

    def forward_conn(self) -> None:
        # forward connection initialization for input to hidden layer
        for i in self.input_layer:
            num_connections = random.randint(1, len(self.hidden_layer))
            hidden_layer_cpy=self.hidden_layer.copy()
            #print(hidden_layer_cpy)
            for _ in range(num_connections):
                choice=random.choice(hidden_layer_cpy)   
                self.graph.add_edge(i,choice)

                hidden_layer_cpy.remove(choice)

        # forward connection initialization for hidden layer to output layer
        for i in self.hidden_layer:
            num_connections = random.randint(1, len(self.output_layer) )
            op_layer_cpy=self.output_layer.copy()
            #print(op_layer_cpy)
            for _ in range(num_connections):
                choice=random.choice(op_layer_cpy)   
                self.graph.add_edge(i,choice)
                op_layer_cpy.remove(choice)

        # forward calculations for output of nn
    def forward_pass(self, inputs) -> list:
        outputs = []
        hidden_outputs = []
        input_to_hidden = []
        hidden_to_hidden = []

        # forward pass for input to hidden layer
        for i in self.input_layer:
            input_to_hidden.append([self.graph.get_value(i)*inputs[i]])
        
        #hidden layer output
        for i in self.hidden_layer:
            back_connections=self.graph.get_back_connections(i)
            #print("back conn",back_connections)
            temp=[]
            for j in back_connections:
                if j in self.input_layer:    
                    for x in input_to_hidden[self.input_layer.index(j)]:
                        temp.append(self.graph.get_value(i)*x)
            temp=sum(temp)
            hidden_to_hidden.append([temp])
        #print(hidden_to_hidden)

        
        if self.hidden_layer_2!=[]:
            for i in self.hidden_layer_2:
                back_connections=self.graph.get_back_connections(i)
                temp=[]
                for j in back_connections:
                    if j in self.input_layer:
                        for x in input_to_hidden[self.input_layer.index(j)]:
                            temp.append(self.graph.get_value(i)*x)
                    elif j in self.hidden_layer:
                        for x in hidden_to_hidden[self.hidden_layer.index(j)]:
                            temp.append(self.graph.get_value(i)*x)
                    else:
                        pass
                temp=sum(temp)
                hidden_outputs.append([temp])

             # outputs    
            for i in self.output_layer:
                back_connections=self.graph.get_back_connections(i)
                temp=[]
                for j in back_connections:
                    if j in self.input_layer:
                        for x in input_to_hidden[self.input_layer.index(j)]:
                            temp.append(self.graph.get_value(i)*x)
                    elif j in self.hidden_layer:
                        for x in hidden_to_hidden[self.hidden_layer.index(j)]:
                            temp.append(self.graph.get_value(i)*x)
                    else:
                        pass
                temp=sum(temp)
                outputs.append([temp])


        else:           
        #final output
            for i in self.output_layer:
                back_connections=self.graph.get_back_connections(i)
                temp=[]
                for j in back_connections:
                    if j in self.input_layer:
                        for x in input_to_hidden[self.input_layer.index(j)]:
                            temp.append(self.graph.get_value(i)*x)
                    elif j in self.hidden_layer:
                        for x in hidden_to_hidden[self.hidden_layer.index(j)]:
                            temp.append(self.graph.get_value(i)*x)
                    else:
                        pass
                temp=sum(temp)
                outputs.append([temp])
            print(outputs) 
                
        return outputs
    

    def add_node(self) :
        if random.randint(0,2)==1:
           # print("added node",self.hidden_layer_2) 
            self.hidden_layer_2.append(self.create_node()) 
            #print("added node",self.hidden_layer_2)
            if random.randint(0,2)==1:
                for i in range(random.randint(2,len(self.hidden_layer))):
                    target_node=random.choice(self.hidden_layer)
             #       print("target node",target_node)
                    self.graph.add_edge(target_node,self.hidden_layer_2[-1])
            else:
                for i in range(random.randint(1,len(self.input_layer))):
                    target_node=random.choice(self.input_layer)
              #      print("target node",target_node)
                    self.graph.add_edge(target_node,self.hidden_layer_2[-1])  

    #better to implement this
    def remove_unconnected_nodes(self) -> None:
        for node in self.graph.nodes:
            #chech logic ones
            
            if len(self.graph.get_back_connections(node))==0 and node not in self.input_layer:
                self.graph.remove_node(node)


"""
The `GeneticAlgorithm` class is responsible for managing the evolution of a population of genomes using a genetic algorithm. It provides methods for evaluating the fitness of genomes, selecting parents, performing crossover and mutation, and returning the best-fitted genome.

The class has the following methods:

- `__init__(self, population_size, input_size, output_size, generation_count)`: Initializes the `GeneticAlgorithm` instance with the specified population size, input size, output size, and generation count.
- `evaluate_fitness(self, input_array)`: Evaluates the fitness of each genome in the population using the provided input array.
- `fitness_function(self, genome, input_array)`: Calculates the fitness of a given genome using the provided input array. This method can be overridden to implement a custom fitness function.
- `select_parents(self, population, k)`: Selects `k` parents from the population based on their fitness.
- `species_managment(self, population)`: Manages the species of the population. This method can be implemented to handle species-based evolution.
- `crossover(self, parent1, parent2)`: Performs crossover between two parent genomes to create a child genome.
- `mutate(self, genome)`: Applies mutation to a given genome.
- `best_fitted_genome(self)`: Returns the best-fitted genome in the population.
- `evolve(self, input_array)`: Evolves the population over multiple generations and returns the best-fitted genome.
- `check_convergence(population)`: Checks if the population has converged based on the best fitness.
"""
class GeneticAlgorithm:
    #perfect
    def __init__(self, population_size, input_size, output_size,generation_count) -> None:
        self.generation_count = generation_count
        self.population = [genome(input_size, output_size,constr=0) for _ in range(population_size)]

    # extension of fitness function 
    #no need to change
    def evaluate_fitness(self,input_array) -> None:
        for genome in self.population:
            genome.fitness = self.fitness_function(genome,input_array)

    #easy function just use forward pass take output and calculate fitness based on performance
    def fitness_function(self, genome,input_array=[1]) -> int:  
        outputs = genome.forward_pass(input_array)
        #remaining function depends on environment
        #write accordingly
        return random.randint(0,100)

    #selects k parents from population based on fitness
    #no need to change this function(selection criteria can be changed)
    def select_parents(self, population, k) -> list:
        parents = []
        for _ in range(k):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            selected_parent = parent1 if parent1.fitness > parent2.fitness else parent2
            parents.append(selected_parent)

        return parents
    
    #it is good to implement this function for proper working of algo
    #however not required
    def species_managment(self, population) -> None:    #code can work without this function
        #complete this function
        pass

    #crossover between 2 parents
    #cross over should be between similar structures check the similarity of structures and then cross over
    #node should be added in prob of 50% after crossover
    def crossover(self,parent1,parent2)-> genome:
        child = genome(len(parent1.input_layer), len(parent1.output_layer))
        winner = parent1 if parent1.fitness > parent2.fitness else parent2
        # Inherit connections from both parents
        for node_id in range(0, min(len(parent1.graph.nodes),len(parent2.graph.nodes))-1):
            if random.random() < 0.5:  # 50% probability of inheriting from each parent
                child.graph.nodes[node_id] = parent1.graph.nodes[node_id]
                if node_id in parent1.input_layer:
                    child.input_layer.append(node_id)
                elif node_id in parent1.hidden_layer:
                    child.hidden_layer.append(node_id)
                elif node_id in parent1.hidden_layer_2:
                        child.hidden_layer_2.append(node_id)

            else:
                child.graph.nodes[node_id] = parent2.graph.nodes[node_id]
                if node_id in parent2.input_layer:
                    child.input_layer.append(node_id)
                elif node_id in parent2.hidden_layer:
                    child.hidden_layer.append(node_id)
                elif node_id in parent2.hidden_layer_2:
                        child.hidden_layer_2.append(node_id)
        
        #insures output node
        node_id=min(len(parent1.graph.nodes),len(parent2.graph.nodes))-1
        if random.random() < 0.5:  # 50% probability of inheriting from each parent 
            child.graph.nodes[node_id] = parent1.graph.nodes[node_id]
            child.output_layer.append(node_id)  
        else:
            child.graph.nodes[node_id] = parent2.graph.nodes[node_id]
            child.output_layer.append(node_id)

        return child

    #function add some variations in childs like adding new node or adding new connection
    #adjust frequency of mutation
    def mutate(self,genome)-> genome :
        if random.random() < 0.1:
            genome.add_node()
        # Add a new connection with a probability of 0.2
        if random.random() < 0.2:
            # Select a random source and target node
            source_node = random.choice(list(genome.graph.nodes.keys()))
            target_node = random.choice(list(genome.graph.nodes.keys()))
            # Add the connection if it doesn't exist and it's not a self-loop
            if target_node != source_node and target_node not in genome.graph.nodes[source_node].connections:
                genome.graph.add_edge(source_node, target_node)
        # Remove a connection with a probability of 0.1
        if random.random() < 0.1:
            # Select a random source and target node with an existing connection
            source_node = random.choice(list(genome.graph.nodes.keys()))
            if genome.graph.nodes[source_node].connections:
                target_node = random.choice(list(genome.graph.nodes[source_node].connections))
                genome.graph.nodes[source_node].connections.remove(target_node)
                if target_node in genome.graph.nodes:
                    #print("exist",target_node)
                    if source_node in genome.graph.nodes[target_node].back_connections:
                        genome.graph.nodes[target_node].back_connections.remove(source_node)

        return genome
    
    #returns best genome
    #no need to change perfect
    def best_fitted_genome(self) -> genome:  
        return max(self.population, key=lambda x: x.fitness)

    #evolves the nn
    #no need to change perfect
    #remove the commented part when commented function works properly 
    def evolve(self,input_array)-> genome:
        while self.generation_count > 0:
            self.generation_count -= 1
            self.evaluate_fitness([1]) 
            parents=self.select_parents(self.population, 10)
            self.population=[]
            print("generation",self.generation_count)
            for _ in range(100):
                parent1, parent2 = random.sample(parents, 2)
                child=self.crossover(parent1, parent2)
                child=self.mutate(child)
                #child.remove_unconnected_nodes()
                self.population.append(child)
                
        return self.best_fitted_genome()
    
    #checking the convergence of the population for algorithm to stop
    #no need to change perfect
    def check_convergence(self,population):
        best_fitness = max(genome.fitness for genome in population)
        if best_fitness >= 90:  
            return True
        return False
    

"""
Evolves a population of genomes using a genetic algorithm approach. The `evolve` method runs the genetic algorithm for the specified number of generations, evaluating the fitness of each genome, selecting parents, and creating new child genomes through crossover and mutation.
    
Args:
    input_array (list): The input data to be used for evaluating the fitness of the genomes.
    
Returns:
        genome: The best-fitted genome after the specified number of generations.
"""

ga = GeneticAlgorithm(population_size=100, input_size=1, output_size=1,generation_count=100)
ga.evolve(input_array=[1])

"""
Graph Class: 
Additionally, the remove_node method does not handle the removal of connections correctly.


Genome Class: 
The remove_unconnected_nodes method is commented out and not implemented.


GeneticAlgorithm Class:
The crossover method inherits connections from both parents but does not handle the case where the number of nodes in the parents' graphs 
does not match. 
add code for changing values of node in either mutation or crossover
"""