import json

class MDP:
    """A Markov Decision Process object with the following internal variables:
    - states (list[str]): a list of state names
    - rewards (dict[str,dict[str,list[float]]]): a dictionary of rewards by state/action/successor-state
    - action_lists (dict[str,list[str]]): a dictionary of an available action list at each state
    - gamma (float): a discount factor
    - transitions (dict[str,dict[str,list[float]]]): A transition matrix: (state, action) -> a list of probabilities to each state (Hint: refer to mdp jason files) 
    """
    def __init__(self, config_path: str) -> None:
        """load the MDP settings from a json config file
        Hint: Use json.load() to load the config file in the dictionary format

        Args:
            config_path (str): a path to the json config file

        Raises:
            ValueError: If a transition probability is invalid (the sum not equal to 1.0), raise an error. (Use __verify_probs function)
        """
        try:
            with open(config_path, 'r') as f:
                jsonobj = json.load(f)
                self.__rewards = jsonobj["rewards"]
                self.__transitions = jsonobj["tran_prob"]
                self.gamma = jsonobj["gamma"]
        except OSError:
            print('An invalid config file path was given.')

        self.__states = self.__transitions.keys()

        self.action_lists = dict()
        for s in self.__states:
            self.action_lists[s] = self.__transitions[s].keys()
        
        self.__verify_probs(self.__transitions)
    
    def __verify_probs(self, trans: dict[str,dict[str,list[float]]]):
        """raise ValueError if the transition matrix has invalid values. In particular, check that the sum of probabilities from a state to successor states is always 1.0.

        Args:
            trans (dict[str,dict[str,list[float]]]): transition matrix (transitions loaded from the json file)

        Raises:
            ValueError: If a transition probability is invalid, raise an error.

        Note:
            This function should be called once when a json file is loaded in __init__().
        """
        for s in self.states():
            for a in self.actions(s):
                tran_probs = trans[s][a]
                if sum(tran_probs) != 1.0:
                    raise ValueError("Invalid Tran Prob.")

    def states(self) -> list[str]:
        """returns a list of all states

        Returns:
            list[str]: a list of states e.g. ["0", "1", "2"]
        """
        return self.__states
    
    def R(self, state: str, action: str, sstate: str) -> float:
        """returns a reward value of a given state

        Args:
            state (str): a state name e.g. "0"
            action (str): an action name e.g. "r"
            successor state (str): a state name e.g. "0"

        Returns:
            float: a reward value
        """
        return self.__rewards[state][action][int(sstate)]
    
    def T(self, state: str, action: str) -> list:
        """returns a list of probabilities to all possible successor states

        Args:
            state (str): a state name e.g. "0"
            action (str): an action e.g. "r"

        Returns:
            list[(str,float)]: a list of probabilities to all possible successor states e.g. [("0", 0.1), ("2", 0.3), ("3", 0.6)] 
        """
        tran_probs = self.__transitions[state][action]

        s_ps = list()
        for s, p in zip(self.__states, tran_probs):
            s_ps.append((s,p))
        return s_ps
    
    def actions(self, state: str) -> list[str]:
        """returns a list of possible actions at a given state

        Args:
            state (str): a state name e.g. "0"

        Returns:
            list[str]: a list of possible actions e.g. ["r", "l"]
        """
        return self.action_lists[state]
