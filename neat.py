
#|***************************************************************************|
#|  add functions like species management for proper NEAT algorithm          |
#|  change crossover and mutation function for proper evolution              |
#|  check edge case in add node function and check if childs are all copy    |
#|  of parents in crossover function                                         |
#|  if more parents are selected all childrens become copy of each other     |
#|  after some generations.optimize forward pass                             |
#|___________________________________________________________________________|

import random
class Node:
    def __init__(self, node_id, connections=[], back=[]) -> None:
        self.id = node_id
        #value_range=(-0.5, 0.5)
        #self.value = random.uniform(*value_range)
        self.connections = []
        self.back_weights=[]
        self.back_connections = []

class graph:
    def __init__(self) -> None:
        self.nodes = {}  

    def add_node(self, node_id) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, [],[])
    
    def get_weight(self,node_id,j) -> int:
        if node_id in self.nodes:
                return self.nodes[node_id].back_weights[j]
        else :
            pass
            

    def add_edge(self, node_id1, node_id2) -> None:
        if node_id2 not in self.nodes[node_id1].connections and node_id1 not in self.nodes[node_id2].back_connections:
            self.nodes[node_id1].connections.append(node_id2)
            self.nodes[node_id2].back_connections.append(node_id1)
            self.nodes[node_id2].back_weights.append(random.random())
        elif node_id2 not in self.nodes[node_id1].connections and node_id1 in self.nodes[node_id2].back_connections:
                self.nodes[node_id1].connections.appeend(node_id2)  # can modify it to add or remove the connection randomly
        elif node_id1 not in self.nodes[node_id2].back_connections and node_id2 in self.nodes[node_id1].connections:#check importance of this statement
                self.nodes[node_id2].back_connections.append(node_id1)
                self.nodes[node_id2].back_weights.append(random.random())
        else :
            pass
            #
    
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
                
                self.nodes.pop(node_id)
            else :
                pass
                

class genome:
    def __init__(self, ip,op,constr=1) -> None:
        if constr==0:
            self.id = 0
            self.graph=graph()
            self.input_layer=[]
            self.fitness = 0
    
            for _ in range(ip):
                self.input_layer.append(self.create_node())

            self.hidden_layer=[]
            for _ in range(random.randint(2, 5)):
                self.hidden_layer.append(self.create_node())
        
            self.hidden_layer_2=[]

            self.output_layer=[]
            for _ in range(op):
                self.output_layer.append(self.create_node())
        
            self.forward_conn()
            self.add_node()
        
            self.remove_unconnected_nodes()
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
            input_to_hidden.append([1*inputs[i]])
        
        #hidden layer output
        for i in self.hidden_layer:
            back_connections=self.graph.get_back_connections(i)
            #print("back conn",back_connections)
            temp=[]
            for j in back_connections:
                if j in self.input_layer:    
                    for x in input_to_hidden[self.input_layer.index(j)]:
                        temp.append(self.graph.get_weight(i,back_connections.index(j))*x)
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
                            temp.append(self.graph.get_weight(i,back_connections.index(j))*x)
                    elif j in self.hidden_layer:
                        for x in hidden_to_hidden[self.hidden_layer.index(j)]:
                            temp.append(self.graph.get_weight(i,back_connections.index(j))*x)
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
                            temp.append(self.graph.get_weight(i,back_connections.index(j))*x)
                    elif j in self.hidden_layer:
                        for x in hidden_to_hidden[self.hidden_layer.index(j)]:
                            temp.append(self.graph.get_weight(i,back_connections.index(j))*x)
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
                            temp.append(self.graph.get_weight(i,back_connections.index(j))*x)
                    elif j in self.hidden_layer:
                        for x in hidden_to_hidden[self.hidden_layer.index(j)]:
                            temp.append(self.graph.get_weight(i,back_connections.index(j))*x)
                    else:
                        pass
                temp=sum(temp)
                outputs.append([temp])
            print(outputs) 
                
        return outputs
    

    def add_node(self) :
        if random.randint(0,2)==1:
            
            self.hidden_layer_2.append(self.create_node()) 
            
            if random.randint(0,2)==1 :
                for i in range(random.randint(1,len(self.hidden_layer))): # 1 can create problem check that case 
                    target_node=random.choice(self.hidden_layer)
                    self.graph.add_edge(target_node,self.hidden_layer_2[-1])
            else:
                for i in range(random.randint(1,len(self.input_layer))):
                    target_node=random.choice(self.input_layer)
                    self.graph.add_edge(target_node,self.hidden_layer_2[-1])  

    
    def remove_unconnected_nodes(self) -> None:
        node_to_remove=[]
        for node in self.graph.nodes:
            if len(self.graph.get_back_connections(node))==0 and node not in self.input_layer:
                node_to_remove.append(node)
        for node in node_to_remove:
            if node in self.hidden_layer_2 :
                self.hidden_layer_2.remove(node)
            elif node in self.hidden_layer:
                self.hidden_layer.remove(node)
            elif node in self.output_layer:
                self.output_layer.remove(node)
            self.graph.remove_node(node)
            

class GeneticAlgorithm:
    
    def __init__(self, population_size, input_size, output_size,generation_count) -> None:
        self.generation_count = generation_count
        self.population = [genome(input_size, output_size,constr=0) for _ in range(population_size)]

    
    def evaluate_fitness(self,input_array) -> None:
        for genome in self.population:
            genome.fitness = self.fitness_function(genome,input_array)

    
    def fitness_function(self, genome,input_array=[1]) -> int:  
        outputs = genome.forward_pass(input_array)
        #remaining function depends on environment
        #write accordingly
        return outputs

   
    def select_parents(self, population, k) -> list:
        parents = []
        for _ in range(k):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            selected_parent = parent1 if parent1.fitness > parent2.fitness else parent2
            parents.append(selected_parent)

        return parents
    
    
    def species_managment(self, population) -> None:    #code can work without this function
        #complete this function
        pass

    
    def crossover(self,parent1,parent2)-> genome:
        child = genome(len(parent1.input_layer), len(parent1.output_layer))
        winner = parent1 if parent1.fitness > parent2.fitness else parent2
        """# Inherit connections from both parents
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
            child.output_layer.append(node_id)"""

        return winner


    def mutate(self,genome)-> genome :
        if random.random() < 0.1:
            genome.add_node()
        
        if random.random() < 0.2:
            source_node = random.choice(list(genome.graph.nodes.keys()))
            target_node = random.choice(list(genome.graph.nodes.keys()))
            if target_node != source_node and target_node not in genome.graph.nodes[source_node].connections:
                genome.graph.add_edge(source_node, target_node)
        
        if random.random() < 0.1:
            source_node = random.choice(list(genome.graph.nodes.keys()))
            if genome.graph.nodes[source_node].connections:
                target_node = random.choice(list(genome.graph.nodes[source_node].connections))
                genome.graph.nodes[source_node].connections.remove(target_node)
                if target_node in genome.graph.nodes:
                    if source_node in genome.graph.nodes[target_node].back_connections:
                        genome.graph.nodes[target_node].back_connections.remove(source_node)
        
        #add proper code for this
        """if random.random() < 0.01:
            genome.remove_node(node_id)
            also remove the connection from the previous node and back connection from next node"""

        return genome
    

    def best_fitted_genome(self) -> genome:  
        return max(self.population, key=lambda x: x.fitness)

    
    def check_convergence(self,population):
        best_fitness = max(genome.fitness for genome in population)
        if best_fitness >= 9:  
            return True
        return False
    

    def evolve(self,input_array)-> genome:
        while self.generation_count > 0:
            self.generation_count -= 1
            self.evaluate_fitness([1]) 
            print(self.population[0].fitness)
            parents=self.select_parents(self.population, 10)
            self.population=[]
            print("generation",self.generation_count)
            for _ in range(100):
                parent1, parent2 = random.sample(parents, 2)
                child=self.crossover(parent1, parent2)
                child=self.mutate(child)
                child.remove_unconnected_nodes()
                self.population.append(child)
                
        return self.best_fitted_genome()
    

ga = GeneticAlgorithm(population_size=10, input_size=1, output_size=1,generation_count=3)
t=ga.evolve(input_array=[1])
print(t.output_layer)
