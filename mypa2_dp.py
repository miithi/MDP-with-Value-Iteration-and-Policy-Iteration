from mymdp import MDP
import math
import json

class MDP:
    def __init__(self, json_file: str):
        with open(json_file) as f:
            data = json.load(f)
        self.tran_prob = data["tran_prob"]
        self.rewards = data["rewards"]
        self.gamma = data["gamma"]
        self.states = list(self.tran_prob.keys())
        self.actions = list(self.tran_prob[self.states[0]].keys())  # Assuming actions are the same for all states

    def get_transition_prob(self, state: str, action: str) -> list:
        """Get the transition probabilities for a given state and action."""
        return self.tran_prob[state][action]

    def get_rewards(self, state: str, action: str) -> list:
        """Get the rewards for a given state and action."""
        return self.rewards[state][action]

    def get_gamma(self) -> float:
        """Get the discount factor gamma."""
        return self.gamma

class ValueAgent:
    """Value-based Agent template (Used as a parent class for VIAgent and PIAgent)
    An agent should maintain:
    - q table (dict[state,dict[action,q-value]])
    - v table (dict[state,v-value])
    - policy table (dict[state,dict[action,probability]])
    - mdp (An MDP instance)
    - v_update_history (list of the v tables): [Grading purpose only] Every time when you update the v table, you need to append the v table to this list. (include the initial value)
    """    
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization

        Args:
            mdp (MDP): An MDP instance
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.            
        """        
        self.q = dict()
        self.v = dict()
        self.pi = dict()
        self.mdp = mdp
        self.thresh = conv_thresh
        self.v_update_history = list()

    def init_random_policy(self):
        """Initialize the policy function with equally distributed random probability.

        When n actions are available at state s, the probability of choosing an action should be 1/n.
        """        
        for state in self.mdp.states:
            self.pi[state] = {}
            actions = self.mdp.actions
            prob = 1 / len(actions)
            for action in actions:
                self.pi[state][action] = prob
                    
    def computeq_fromv(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Given a state-value table, compute the action-state values.
        For deterministic actions, q(s,a) = E[r] + v(s'). Check the lecture slides.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            dict[str,dict[str,float]]: a q value table {state:{action:q-value}}
        """
        q = {}
        for state in self.mdp.states:
            q[state] = {}
            for action in self.mdp.actions:
                q_value = 0
                for next_state, prob in enumerate(self.mdp.get_transition_prob(state, action)):
                    reward = self.mdp.get_rewards(state, action)[next_state]
                    q_value += prob * (reward + self.mdp.get_gamma() * v[str(next_state)])
                q[state][action] = q_value
        return q

    def greedy_policy_improvement(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Greedy policy improvement algorithm. Given a state-value table, update the policy pi.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        q = self.computeq_fromv(v)
        new_pi = {}
        for state in self.mdp.states:
            best_action = max(q[state], key=q[state].get)  # Find the action with the maximum Q-value
            new_pi[state] = {action: 1.0 if action == best_action else 0.0 for action in self.mdp.actions}
        return new_pi

    def check_term(self, v: dict[str,float], next_v: dict[str,float]) -> bool:
        """Return True if the state value has NOT converged.
        Convergence here is defined as follows: 
        For ANY state s, the update delta, abs(v'(s) - v(s)), is within the threshold (self.thresh).

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}
            next_v (dict[str,float]): a state value table (after update)

        Returns:
            bool: True if continue; False if converged
        """
        for state in self.mdp.states:
            if abs(next_v[state] - v[state]) > self.thresh:
                return True  # Keep iterating if the change is greater than the threshold
        return False     


class PIAgent(ValueAgent):
    """Policy Iteration Agent class
    """    
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """        
        super().__init__(mdp, conv_thresh)
        super().init_random_policy() # initialize its policy function with the random policy

    def __iter_policy_eval(self, pi: dict[str,dict[str,float]]) -> dict[str,float]:
        """Iterative policy evaluation algorithm. Given a policy pi, evaluate the value of states (v).

        This function should be called in policy_iteration().

        Args:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}

        Returns:
            dict[str,float]: state-value table {state:v-value}
        """
        V = {state: 0 for state in self.mdp.states}  # Initialize v table
    
        while True:
            next_v = V.copy()
            for state in self.mdp.states:
                value = 0
                for action, prob in pi[state].items():
                    q_value = 0
                    for next_state, trans_prob in enumerate(self.mdp.get_transition_prob(state, action)):
                        reward = self.mdp.get_rewards(state, action)[next_state]
                        q_value += trans_prob * (reward + self.mdp.get_gamma() * V[str(next_state)])
                    value += prob * q_value
                next_v[state] = value
            
            # Check for convergence
            if not self.check_term(V, next_v):
                break
            
            V = next_v
        return V


    def policy_iteration(self) -> dict[str,dict[str,float]]:
        """Policy iteration algorithm. Iterating iter_policy_eval and greedy_policy_improvement, update the policy pi until convergence of the state-value function.

        This function is called to run PI. 
        e.g.
        mdp = MDP("./mdp1.json")
        dpa = PIAgent(mdp)
        dpa.policy_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        self.v_update_history.append(self.v.copy())  # Save initial V table
    
        while True:
            # Policy evaluation
            self.v = self.__iter_policy_eval(self.pi)
            self.v_update_history.append(self.v.copy())  # Save updated V table
            
            # Policy improvement
            new_pi = self.greedy_policy_improvement(self.v)
            
            if new_pi == self.pi:  # If policy is unchanged, break
                break
            
            self.pi = new_pi  # Update policy
        return self.pi


class VIAgent(ValueAgent):
    """Value Iteration Agent class
    """
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """        
        super().__init__(mdp, conv_thresh)
        super().init_random_policy() # initialize its policy function with the random policy

    def value_iteration(self) -> dict[str,dict[str,float]]:
        """Value iteration algorithm. Compute the optimal v values using the value iteration. After that, generate the corresponding optimal policy pi.

        This function is called to run VI. 
        e.g.
        mdp = MDP("./mdp1.json")
        via = VIAgent(mdp)
        via.value_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        V = {state: 0 for state in self.mdp.states}  # Initialize 
    
        self.v_update_history.append(V.copy())  # Save initial v table
        
        while True:
            next_v = V.copy()
            for state in self.mdp.states:
                action_values = []
                for action in self.mdp.actions:
                    q_value = 0
                    for next_state, prob in enumerate(self.mdp.get_transition_prob(state, action)):
                        reward = self.mdp.get_rewards(state, action)[next_state]
                        q_value += prob * (reward + self.mdp.get_gamma() * V[str(next_state)])
                    action_values.append(q_value)
                next_v[state] = max(action_values)
            
            self.v_update_history.append(next_v.copy())  # Save the updated V table
            
            # Check for convergence
            if not self.check_term(V, next_v):
                break
            
            V = next_v
        
        # Update the policy using the final value function
        self.pi = self.greedy_policy_improvement(V)
        return self.pi
    

if __name__ == '__main__':
    mdp = MDP('mdp1.json')
    
    print("Running Value Iteration:")
    vi_agent = VIAgent(mdp)
    vi_policy = vi_agent.value_iteration()
    print("Optimal Policy (Value Iteration):", vi_policy)
    
    # Value Iteration history
    with open("vi_history.log", "w") as vi_log_file:
        for idx, v in enumerate(vi_agent.v_update_history):
            vi_log_file.write(f"Update {idx}: {v}\n")

    print("\nRunning Policy Iteration:")
    pi_agent = PIAgent(mdp)
    pi_policy = pi_agent.policy_iteration()
    print("Optimal Policy (Policy Iteration):", pi_policy) 

    # Policy Iteration history
    with open("pi_history.log", "w") as pi_log_file:
        for idx, v in enumerate(pi_agent.v_update_history):
            pi_log_file.write(f"Update {idx}: {v}\n")
