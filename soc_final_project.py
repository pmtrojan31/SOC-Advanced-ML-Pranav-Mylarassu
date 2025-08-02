import json
from collections import defaultdict, deque
import heapq
from itertools import product
import math

########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################

class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        
        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.
        
        Refer to the sample test case for the structure of the input data.
        """
        self.num_variables = data['num_variables']
        self.variables = list(range(self.num_variables))
        self.variable_domains = data['variable_domains']
        self.edges = data['edges']
        self.potentials = data['potentials']
        self.k = data['k']
        
        # We create a dictionary containing each node as keys and the others nodes its connected to as values
        self.graph = defaultdict(set)
        for edge in self.edges:
            u, v = edge
            self.graph[u].add(v)
            self.graph[v].add(u)
        
        self.cliques = []
        self.junction_tree = defaultdict(set)
        self.clique_potentials = {}
        self.separators = {}
        
    def triangulate_and_get_cliques(self):
        """
        Triangulate the undirected graph and extract the maximal cliques.
        
        What to do here:
        ----------------
        - Implement the triangulation algorithm to make the graph chordal.
        - Extract the maximal cliques from the triangulated graph.
        - Store the cliques for later use in junction tree creation.

        Refer to the problem statement for details on triangulation and clique extraction.
        """
        # We'll create a copy of the graph for triangulation
        triangulated_graph = defaultdict(set)
        for v in self.graph:
            triangulated_graph[v] = self.graph[v].copy()
        
        # Variables that are isolated wont be in self.graph. But we must add them to the triangulated graph.
        for v in self.variables:
            if v not in triangulated_graph:
                triangulated_graph[v] = set()
        
        # We use the minimum fill-in method to decide the variable order of elimination
        eliminated = set()
        elimination_order = []
        
        while len(eliminated) < self.num_variables:
            # To find variable with minimum fill-in edges
            min_fill = float('inf')
            min_var = None
            
            for v in self.variables:
                if v in eliminated:
                    continue
                
                # Count fill-in edges needed
                neighbors = [n for n in triangulated_graph[v] if n not in eliminated]
                fill_edges = 0
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if neighbors[j] not in triangulated_graph[neighbors[i]]:
                            fill_edges += 1
                
                if fill_edges < min_fill:
                    min_fill = fill_edges
                    min_var = v
            
            # We then eliminate the variable
            eliminated.add(min_var)
            elimination_order.append(min_var)
            
            # Then add fill-in edges
            neighbors = [n for n in triangulated_graph[min_var] if n not in eliminated]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    triangulated_graph[neighbors[i]].add(neighbors[j])
                    triangulated_graph[neighbors[j]].add(neighbors[i])
        
        # Now we extract maximal cliques using elimination order
        self.cliques = []
        for var in elimination_order:
            # Start off the clique with just the given variable in the elimination order
            clique = {var}
            for neighbor in self.graph[var]:
                if neighbor in elimination_order[elimination_order.index(var):]:
                    clique.add(neighbor)
            
            # Add edges that were present when var was eliminated
            for other_var in elimination_order[elimination_order.index(var) + 1:]:
                if other_var in triangulated_graph[var]:
                    clique.add(other_var)
            
            if len(clique) > 1:
                clique = sorted(list(clique))
                if clique not in self.cliques:
                    self.cliques.append(clique)
        
        # We then add single variable cliques
        for var in self.variables:
            found_in_clique = False
            for clique in self.cliques:
                if var in clique:
                    found_in_clique = True
                    break
            if not found_in_clique:
                self.cliques.append([var])

    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.
        
        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.

        Refer to the problem statement for details on junction tree construction.
        """
        if len(self.cliques) <= 1:
            return
        
        # We create a complete list of pairs of cliques along with the variables that are common among them, along with their number.
        clique_graph = []
        for i in range(len(self.cliques)):
            for j in range(i + 1, len(self.cliques)):
                separator = set(self.cliques[i]).intersection(set(self.cliques[j]))
                if separator:
                    weight = len(separator) 
                    clique_graph.append((weight, i, j, separator))
        
        # Sort by weight (separator size) in descending order
        clique_graph.sort(reverse=True)
        
        # We use this thing called Kruskal's algorithm to find the "maximum spanning tree"
        parent = list(range(len(self.cliques)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        # Build the junction tree
        self.junction_tree = defaultdict(set)
        self.separators = {}
        
        for weight, i, j, separator in clique_graph:
            if union(i, j):
                self.junction_tree[i].add(j)
                self.junction_tree[j].add(i)
                self.separators[(i, j)] = separator
                self.separators[(j, i)] = separator

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        
        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.
        
        Refer to the sample test case for how potentials are associated with cliques.
        """
        # Initialize all cliques with uniform potentials
        self.clique_potentials = {}
        for i, clique in enumerate(self.cliques):
            # Create potential table for this clique
            domain_sizes = [self.variable_domains[var] for var in clique]
            total_assignments = 1
            for size in domain_sizes:
                total_assignments *= size
            
            # Initialize with ones
            potential_table = {}
            for assignment in product(*[range(self.variable_domains[var]) for var in clique]):
                key = tuple(zip(clique, assignment))
                potential_table[key] = 1.0
            
            self.clique_potentials[i] = potential_table
        
        # Assign given potentials to appropriate cliques
        for potential in self.potentials:
            variables = potential['variables']
            values = potential['values']
            
            # Find a clique that contains all variables in this potential
            assigned = False
            for i, clique in enumerate(self.cliques):
                if all(var in clique for var in variables):
                    # Multiply this potential into the clique
                    for assignment_idx, value in enumerate(values):
                        # Convert flat index to variable assignment
                        assignment = []
                        temp_idx = assignment_idx
                        for var in reversed(variables):
                            assignment.append(temp_idx % self.variable_domains[var])
                            temp_idx //= self.variable_domains[var]
                        assignment.reverse()
                        
                        # Create full clique assignment
                        clique_assignment = {}
                        for j, var in enumerate(variables):
                            clique_assignment[var] = assignment[j]
                        
                        # Find all compatible clique assignments
                        for key in self.clique_potentials[i]:
                            compatible = True
                            for var, val in clique_assignment.items():
                                if any(k[0] == var and k[1] != val for k in key):
                                    compatible = False
                                    break
                            
                            if compatible:
                                self.clique_potentials[i][key] *= value
                    
                    assigned = True
                    break
            
            if not assigned:
                # Create a new clique for this potential if needed
                clique = sorted(variables)
                self.cliques.append(clique)
                clique_idx = len(self.cliques) - 1
                
                potential_table = {}
                for assignment_idx, value in enumerate(values):
                    assignment = []
                    temp_idx = assignment_idx
                    for var in reversed(variables):
                        assignment.append(temp_idx % self.variable_domains[var])
                        temp_idx //= self.variable_domains[var]
                    assignment.reverse()
                    
                    key = tuple(zip(variables, assignment))
                    potential_table[key] = value
                
                self.clique_potentials[clique_idx] = potential_table

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.
        
        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.
        
        Refer to the problem statement for details on computing the partition function.
        """
        if not self.cliques:
            return 1.0
        
        if len(self.cliques) == 1:
            return sum(self.clique_potentials[0].values())
        
        # Choose root clique
        root = 0
        
        # Collect messages from leaves to root
        messages = {}
        visited = set()
        
        def collect_messages(clique_idx, parent_idx):
            visited.add(clique_idx)
            
            for neighbor in self.junction_tree[clique_idx]:
                if neighbor != parent_idx and neighbor not in visited:
                    collect_messages(neighbor, clique_idx)
                    
                    # Compute message from neighbor to clique_idx
                    separator = self.separators.get((neighbor, clique_idx), set())
                    
                    if separator:
                        message = self.compute_message(neighbor, clique_idx, separator)
                        messages[(neighbor, clique_idx)] = message
        
        collect_messages(root, -1)
        
        # Compute Z by marginalizing root clique
        root_potential = self.clique_potentials[root].copy()
        
        # Multiply in all received messages
        for neighbor in self.junction_tree[root]:
            if (neighbor, root) in messages:
                message = messages[(neighbor, root)]
                for key in root_potential:
                    # Find corresponding message entry
                    separator_assignment = []
                    separator_vars = sorted(list(self.separators[(neighbor, root)]))
                    for var in separator_vars:
                        for k in key:
                            if k[0] == var:
                                separator_assignment.append((var, k[1]))
                                break
                    separator_key = tuple(separator_assignment)
                    if separator_key in message:
                        root_potential[key] *= message[separator_key]
        
        return sum(root_potential.values())
    
    def compute_message(self, from_clique, to_clique, separator):
        """Helper function to compute message between cliques"""
        potential = self.clique_potentials[from_clique].copy()
        
        # Variables to marginalize out
        from_vars = set(self.cliques[from_clique])
        marginalize_vars = from_vars - separator
        
        if not marginalize_vars:
            # No marginalization needed
            message = {}
            for key, value in potential.items():
                separator_key = tuple((var, val) for var, val in key if var in separator)
                if separator_key in message:
                    message[separator_key] += value
                else:
                    message[separator_key] = value
            return message
        
        # Marginalize out variables not in separator
        message = {}
        for key, value in potential.items():
            separator_assignment = tuple((var, val) for var, val in key if var in separator)
            if separator_assignment in message:
                message[separator_assignment] += value
            else:
                message[separator_assignment] = value
        
        return message

    def compute_marginals(self):
        """
        Compute the marginal probabilities for all variables in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to compute the marginal probabilities for each variable.
        - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.
        
        Refer to the sample test case for the expected format of the marginals.
        """
        z_value = self.get_z_value()
        if z_value == 0:
            # Return uniform distribution for each variable
            marginals = []
            for var in self.variables:
                domain_size = self.variable_domains[var]
                uniform_prob = 1.0 / domain_size
                marginals.append([uniform_prob] * domain_size)
            return marginals
        
        marginals = []
        
        for var in self.variables:
            # Find a clique containing this variable
            clique_idx = None
            for i, clique in enumerate(self.cliques):
                if var in clique:
                    clique_idx = i
                    break
            
            if clique_idx is None:
                # Variable not found in any clique, return uniform
                domain_size = self.variable_domains[var]
                uniform_prob = 1.0 / domain_size
                marginals.append([uniform_prob] * domain_size)
                continue
            
            # Marginalize the clique potential to get variable marginal
            variable_marginal = [0.0] * self.variable_domains[var]
            
            for key, value in self.clique_potentials[clique_idx].items():
                for var_assignment in key:
                    if var_assignment[0] == var:
                        variable_marginal[var_assignment[1]] += value
                        break
            
            # Normalize
            total = sum(variable_marginal)
            if total > 0:
                variable_marginal = [prob / total for prob in variable_marginal]
            else:
                domain_size = self.variable_domains[var]
                variable_marginal = [1.0 / domain_size] * domain_size
            
            marginals.append(variable_marginal)
        
        return marginals

    def compute_top_k(self):
        """
        Compute the top-k most probable assignments in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.
        
        Refer to the sample test case for the expected format of the top-k assignments.
        """
        z_value = self.get_z_value()
        if z_value == 0:
            return []
        
        # We first generate all possible assignments and their probabilities
        all_assignments = []
        
        for assignment in product(*[range(self.variable_domains[var]) for var in self.variables]):
            prob = self.compute_assignment_probability(assignment)
            if prob > 0:
                all_assignments.append((prob, list(assignment)))
        
        # Sort by probability in descending order
        all_assignments.sort(reverse=True)
        
        # Now we return the k combinations with highest probabilities
        top_k = []
        for i in range(min(self.k, len(all_assignments))):
            prob, assignment = all_assignments[i]
            normalized_prob = prob / z_value
            top_k.append({
                'assignment': assignment,
                'probability': normalized_prob
            })
        
        return top_k
    
    def compute_assignment_probability(self, assignment):
        """Helper function to compute probability of a specific assignment"""
        prob = 1.0
        
        for potential in self.potentials:
            variables = potential['variables']
            values = potential['values']
            
            # We get assignment for these variables
            var_assignment = [assignment[var] for var in variables]
            
            # Converting to flat index
            flat_index = 0
            multiplier = 1
            for i in reversed(range(len(variables))):
                flat_index += var_assignment[i] * multiplier
                multiplier *= self.variable_domains[variables[i]]
            
            if flat_index < len(values):
                prob *= values[flat_index]
            else:
                prob *= 0
        
        return prob

########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('Sample_Testcase.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')
