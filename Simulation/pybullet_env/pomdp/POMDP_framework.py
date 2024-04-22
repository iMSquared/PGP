""" 
For defining POMDP problem, this API contain basic classes & functions of POMDP.

[Terminology]
S: state space
A: action space
O: observation space
T: Transition model
Z: observation model
R: reward model

Adapt from the reference: https://github.com/h2r/pomdp-py/tree/master/pomdp_py/framework
"""

from abc import ABC, abstractmethod
import random
from typing import Any, Callable, Dict, Tuple, NamedTuple, Iterator, TypeVar, List
from dataclasses import dataclass
from copy import deepcopy


TerminationT = TypeVar("TerminationT", bound=str)
TERMINATION_FAIL    : TerminationT = "fail"
TERMINATION_SUCCESS : TerminationT = "success"
TERMINATION_CONTINUE: TerminationT = "continue"
PhaseT = TypeVar("PhaseT", bound=str)
PHASE_EXECUTION : PhaseT = "execution"
PHASE_SIMULATION: PhaseT = "simulation"
PHASE_ROLLOUT   : PhaseT = "rollout"


class State(ABC):
    """
    The State Class. State must be `hashable`.
    """

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError


class Action(ABC):
    """
    The Action class. Action must be `hashable`.
    """

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError


class Observation(ABC):
    """
    The Observation class. Observation must be `hashable`.
    """

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError



@dataclass(eq=True, frozen=True)
class HistoryEntry:
    """
    A history entry which records the history of (a, o, r) sequence.
    @dataclass(frozen=True) will define the __eq__() and __hash__().
    """
    action: Action
    observation: Observation
    reward: float
    phase: PhaseT



class TransitionModel(ABC):
    """
    T(s,a,s') = Pr(s'|s,a)
    """

    @abstractmethod
    def probability(self, next_state: State, cur_state: State, action: Action, *args, **kwargs):
        """
        Args:
            next_state(s'): State
            cur_state(s): State
            action(a): Action
        Returns:
            Pr(s'|s,a)
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, state: State, action: Action, *args, **kwargs):
        """
        Returns next state randomly sampled according to the distribution of this transition model.
        
        Args:
            state(s): State
            action(a): Action
        Returns:
            next_state(s'): State, ...
        """
        raise NotImplementedError


class ObservationModel(ABC):
    """
    Z(s',a,o) = Pr(o|s',a). This means perception model, not sensor mesurement.
    """

    @abstractmethod
    def probability(self, observation: Observation, next_state: State, action: Action, *args, **kwargs):
        """
        Args:
            observation(o): Observation
            next_state(s'): State
            action(a): Action
        Returns:
            Pr(o|s',a)
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, next_state: State, action: Action, *args, **kwargs):
        """sample(self, next_state, action, **kwargs)
        Returns observation randomly sampled according to the distribution of this observation model.

        Args:
            next_state(s'): State
            action(a): Action
        Returns:
            observation(o): Observation, ...
        """
        raise NotImplementedError

    @abstractmethod
    def get_sensor_observation(self, next_state: State, action: Action, *args, **kwargs):
        """
        Returns an sensor_observation
        """
        raise NotImplementedError


class RewardModel(ABC):
    """
    R(s,a,s') = Pr(r|s',a)
    """

    @abstractmethod
    def probability(self, reward: float, state: State, action: Action, next_state: State, *args, **kwargs):
        """
        Args:
            reward(r): float
            next_state(s'): State
            action(a): Action
        Returns:
            Pr(r|s',a)
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, state: State, action: Action, next_state: State, *args, **kwargs):
        """
        Returns reward randomly sampled according to the distribution of this reward model.
        Args:
            state(s) <- basics.State
            action(a) <- basics.Action
            next_state(s') <- basics.State
        Returns:
            reword(r): float, termination
        """
        raise NotImplementedError

    @abstractmethod
    def _check_termination(self, state: State) -> bool:
        """
        Check the state satisfy terminate condition
        """
        raise NotImplementedError


class BlackboxModel:
    """
    A BlackboxModel is the generative distribution G(s,a)
    which can generate samples where each is a tuple (s',o,r) according to T, Z, and R.
    (These models can be different to models of environment.)
    """
    def __init__(self,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        reward_model: RewardModel):

        self._transition_model = transition_model
        self._observation_model = observation_model
        self._reward_model = reward_model

    def sample(self, state: State, action: Action):
        """
        Sample (s',o,r) ~ G(s',a)
        """
        # |NOTE(Jiyong)|: how to deal rest information of models
        next_state = self._transition_model.sample(state, action)
        observation = self._observation_model.sample(next_state, action)
        reward, termination = self._reward_model.sample(next_state, action, state)

        return next_state, observation, reward, termination
    
    @property
    def transition_model(self):
        return self._transition_model
    
    @property
    def observation_model(self):
        return self._observation_model
    
    @property
    def reward_model(self):
        return self._reward_model


class PolicyModel(ABC):
    """
    \pi(a|h)
    """

    @abstractmethod
    def sample(self, init_observation: Observation, history: Tuple, state: State, goal: object, *args, **kwargs) -> Action:
        """
        Returns action randomly sampled according to the policy.
        Args:
            init_observation (Observation): Initial observation
            history (Tuple): Tuple of sequece of observations, and actions (o_0, a_1, o_1, (r_1,) a_2, o_2, (r_2,) ...)
            state (State): Current state of a particle.
            goal (goal): Goal condition
        Returns:
            action (Action): Action
        """
        raise NotImplementedError


class RolloutPolicyModel(ABC):

    @abstractmethod
    def sample(self, init_observation: Observation, history: Tuple, state: State, goal: object, *args, **kwargs) -> Action:
        """
        Returns action randomly sampled according to the policy.
        Args:
            init_observation (Observation): Initial observation
            history (Tuple): Tuple of sequece of observations, and actions (o_0, a_1, o_1, (r_1,) a_2, o_2, (r_2,) ...)
            state (State): Current state of a particle.
            goal (goal): Goal condition
        Returns:
            action (Action): Action
        """
        raise NotImplementedError


class ValueModel(ABC):

    def __init__(self, is_v_model: bool):
        """Value model can either be V model or Q model

        Args:
            is_v_model (bool): Mark this as True if the model predicts the value model.
        """
        self.is_v_model = is_v_model

    @abstractmethod
    def sample(self, init_observation: Observation, history: Tuple, *args, **kwargs):
        """
        Returns value of the node using value network or heuristic function.
        Args:
            init_observation (Observation): Initial observation
            history(h): Tuple of sequece of observations, and actions (o_0, a_1, o_1, (r_1,) a_2, o_2, (r_2,) ...)
        Returns:
            value: float
        """
        raise NotImplementedError


class Belief(ABC):
    """
    The Belief class. Pr(s).
    A Distribution is a probability function that maps from state to a real value,
    that is, `Pr(S=s)`.
    """
    
    @abstractmethod
    def sample(self):
        """
        Sample a state based on the underlying belief distribution
        """
        raise NotImplementedError

    @abstractmethod
    def probability(self):
        """
        Returns the probabiltiy of the state.
        """
        raise NotImplementedError


class UnweightedParticles(Belief):
    def __init__(self, particles: Tuple[State]):
        if not isinstance(particles, tuple):
            raise TypeError("particles must be a tuple.")
        self._particles = particles

    @property
    def particles(self):
        return self._particles

    def __len__(self):
        return len(self._particles)
    
    def __getitem__(self, index: int):
        return self._particles[index]

    def __iter__(self):
        """Initialization of iterator over the values in these particles"""
        return iter(self._particles)

    def sample(self):
        """
        Randomly choose a particle
        """
        if len(self._particles) > 0:
            return random.choices(self._particles)[0]
        else:
            return None
    
    def probability(self, state: State):
        """
        Returns the probability of `weight`.
        """   
        if state not in self._particles:
            return 0.
        else:
            return 1/len(self._particles)
        

# |FIXME(Jiyong)|: It can support continuous state particle only. Because it cannot merge same particles.
class WeightedParticles(Belief):
    def __init__(self, particles: Dict[State, float]=None):
        if particles is not None:
            self._particles = particles
        else:
            self._particles = Dict()
        
    @property
    def particles(self):
        return self._particles

    def __len__(self):
        return len(self._particles)
    
    def __getitem__(self, state: State):
        """
        Returns the value of `weight`.
        """        
        return self._particles[state]
    
    def __setitem__(self, state: State, weight: float):
        """
        Set the weight of `state` as `weight`.
        """        
        self._particles[state] = weight

    def __iter__(self):
        """Initialization of iterator over the values in these particles"""
        return iter(self._particles)

    def sample(self):
        """
        Randomly choose a particle
        """
        if len(self._particles) > 0:
            # Handling the division by zero.
            if sum(list(self._particles.values())) != 0:
                return random.choices(list(self._particles.keys()), list(self._particles.values()))[0]
            else:
                return random.choice(list(self._particles.keys()))
        else:
            return None
    
    def argmax(self):
        """
        Choose a particle that is most likely to be sampled.
        """
        max_particle = max(self._particles, key=self._particles.get)
        return max_particle

    def probability(self, state: State):
        """
        Returns the probability of `weight`.
        """   
        if state not in self._particles:
            return 0.
        else:
            hist = self.get_histogram()
            return hist[state]

    def get_histogram(self):
        """
        !!!DEPRECATED WARNING!!!
        Returns a dictionary from value to probability of the histogram
        """        
        hist = {}
        total_val = 0.
        for s, v in zip(self._particles):
            total_val += v
            if s not in hist:
                hist[s] = v
            hist[s] += v
        for s in hist:
            hist[s] = hist[s] / total_val
        return hist


class Agent(ABC):
    """
    An Agent operates in an environment by giving actions, receiving observations and rewards, and updating its belief.
    """
    def __init__(self, blackbox_model: BlackboxModel,
                       policy_model: PolicyModel,
                       rollout_policy_model: RolloutPolicyModel,
                       value_model: ValueModel = None,
                       init_belief: UnweightedParticles = None,
                       init_observation: Observation = None,
                       goal_condition = None):

        self._blackbox_model = blackbox_model
        self._policy_model = policy_model
        self._rollout_policy_model = rollout_policy_model
        self._value_model = value_model
        self._init_belief = init_belief
        self._init_observation = init_observation
        self._goal_condition = goal_condition 

        # For online planning
        self._cur_belief = init_belief
        self._history: Tuple[HistoryEntry] = ()

    # getter & setter
    @property
    def blackbox_model(self):
        return self._blackbox_model

    @property
    def history(self):
        """
        Current history form of ((a_1, o_1, r_1), (a_2, o_2, r_2), ...)
        """
        return self._history

    # |NOTE(Jiyong)|: This is for real history after execution, not for histories during planning. For histories during planning, use history of nodes
    def _update_history(self, real_action: Action, real_observation: Observation, real_reward: float):
        self._history += (HistoryEntry(real_action, real_observation, real_reward, PHASE_EXECUTION),)

    @property
    def init_belief(self):
        """
        Initial belief distribution
        """
        return self._init_belief

    @property
    def init_observation(self):
        """Inital observation"""
        return self._init_observation

    @property
    def goal_condition(self):
        """Goal condition"""
        return self._goal_condition

    @property
    def belief(self):
        """
        Current belief distribution
        """
        return self._cur_belief
    
    def set_belief(self, belief: Belief, prior: bool=False):
        self._cur_belief = belief
        if prior:
            self._init_belief = belief

    def sample_belief(self):
        """
        Returns a state (class:`State`) sampled from the belief.
        """
        return self._cur_belief.sample()

    def get_action(self, history: Tuple, state:State, goal = None, rollout: bool= True):
        if rollout:
            return self._rollout_policy_model.sample(self.init_observation, history, state, goal)
        else:
            return self._policy_model.sample(self.init_observation, history, state, goal)

    def get_value(self, history: Tuple, state:State, goal = None):
        return self._value_model.sample(self.init_observation, history, state, goal)

    def get_weight(self, observation: Observation, next_state: State, action: Action):
        """
        Get the weight of the particle after transition, based on observation likelihood Z(o|s',a,s)
        """
        return self._blackbox_model.observation_model.probability(observation, next_state, action)

    def imagine_state(self, state: State, reset):
        self._set_simulation(state, reset)

    @abstractmethod
    def _set_simulation(self, state: State):
        """
        Set simulation environment with given state.
        It is used to imagine state during planning.
        User should implement according to the simulator they want to use.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, real_action: Action, real_observation: Observation, real_reward: float):
        """
        updates the history and performs belief update
        """
        raise NotImplementedError

    def add_attr(self, attr_name: str, attr_value: Any):
        """
        TODO(ssh): Deprecate all add_attr in this project.
        add_attr(self, attr_name, attr_value)
        A function that allows adding attributes to the agent.
        Sometimes useful for planners to store agent-specific information.
        """
        if hasattr(self, attr_name):
            raise ValueError("attributes %s already exists for agent." % attr_name)
        else:
            setattr(self, attr_name, attr_value)


class Environment:
    """
    An Environment gives an observation and an reward to the agent after receiving an action
    while maintains the true state of the world.
    """
    def __init__(self,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        reward_model: RewardModel,
        init_state: State = None):
        """Simple constructor
        Init with init_state as _cur_state."""
        self._init_state = init_state
        self._transition_model = transition_model
        self._observation_model = observation_model
        self._reward_model = reward_model
        self._cur_state = init_state

    @property
    def state(self):
        """Get current true state"""
        return self._cur_state
    
    def set_state(self, state: State, prior: bool=False):
        """Set _cur_state of the POMDP environment.

        Args:
            state (State): State instance.
            prior (bool, optional): When true, updates the _init_state too. Defaults to False.
        """
        self._cur_state = state
        if prior:
            self._init_state = state

    def execute(self, action: Action):
        """
        Execute action, apply state transition given `action` and return sensor observation, reward, and is_goal.
        (should not return next_state, due to the agent can't know about true state)

        Args:
            action (Action): Action that triggers the state transition.
        Returns:
            observation (Observation): Observation instance
            reward (float): Reward scalar value
            termination (bool): True when success, False when fail and None when not ended.
        """

        # Sampling next_state and reward
        next_state = self._transition_model.sample(self._cur_state, action, True)
        observation = self._observation_model.sample(next_state, action)
        reward, termination = self._reward_model.sample(next_state, action, self._cur_state)

        # Apply state transitioin
        self._cur_state = next_state

        return observation, reward, termination
    
    def check_termination(self, state: State):
        return self._reward_model._check_termination(state)
    
    def add_attr(self, attr_name: str, attr_value: Any):
        """
        add_attr(self, attr_name, attr_value)
        A function that allows adding attributes to the agent.
        Sometimes useful for planners to store agent-specific information.
        """
        if hasattr(self, attr_name):
            raise ValueError("attributes %s already exists for agent." % attr_name)
        else:
            setattr(self, attr_name, attr_value)


class POMDP:
    """
    A POMDP instance <- agent(class:'Agent'), env(class:'Enviroment')

    Because POMDP modelling may be different according to the agent and the environment, 
    it does not make much sense to have the POMDP class to hold this information.
    Instead, Agent should have its own belief, policy and the Environment should have its own T, Z, R.
    """

    def __init__(self, agent: Agent, env: Environment, name: str="POMDP"):
        self.agent = agent
        self.env = env
        self.name = name