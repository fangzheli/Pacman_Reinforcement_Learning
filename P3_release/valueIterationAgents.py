import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for k in range(self.iterations):
            next_iteration = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                Vvalue = max([self.getQValue(state,action) for action in self.mdp.getPossibleActions(state)])
                next_iteration[state] = Vvalue
            self.values = next_iteration

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Qvalue = 0
        for nextstate,p in self.mdp.getTransitionStatesAndProbs(state,action):
            Qvalue += (self.getValue(nextstate)*self.discount + self.mdp.getReward(state,action,nextstate))* p
        return Qvalue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        Qvalues =[self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
        if Qvalues:
            return self.mdp.getPossibleActions(state)[Qvalues.index(max(Qvalues))]
        else:
            return None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for k in range(self.iterations):
            state = self.mdp.getStates()[k % len(self.mdp.getStates())]
            if self.mdp.isTerminal(state):
                continue
            Vvalue = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            self.values[state] = Vvalue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = dict()
        for state in self.mdp.getStates():
            predecessors[state] = set()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for nextstate, p in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[nextstate].add(state)
        States_to_update = util.PriorityQueue()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            MaxQvalue = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            diff = abs(self.values[state] - MaxQvalue)
            States_to_update.push(state,-diff)
        for k in range(self.iterations):
            if States_to_update.isEmpty():
                break
            else:
                state = States_to_update.pop()
                if self.mdp.isTerminal(state):
                    continue
                Vvalue = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
                self.values[state] = Vvalue
                for pre_state in predecessors[state]:
                    if self.mdp.isTerminal(pre_state):
                        continue
                    MaxQvalue = max([self.getQValue(pre_state, action) for action in self.mdp.getPossibleActions(pre_state)])
                    diff = abs(self.values[pre_state] - MaxQvalue)
                    if diff > self.theta:
                        States_to_update.update(pre_state, -diff)
