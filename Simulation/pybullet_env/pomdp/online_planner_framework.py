"""
For solving POMDP problem using online planner with given policy, this API contain basic classes & functions of planner.
"""
from typing import Union, Generic
from pomdp.POMDP_framework import *


class TreeNode(ABC):
    def __init__(self):
        self.children: Dict[Union[Action, Observation], TreeNode] = {}

    def __getitem__(self, key: Union[Action, Observation]) -> "TreeNode":
        if key not in self.children and type(key) == int:
            clist = list(self.children)
            if key >= 0 and key < len(clist):
                return self.children[clist[key]]
            else:
                return None
        return self.children.get(key,None)

    def __setitem__(self, key: Union[Action, Observation], 
                          value: "TreeNode"):
        self.children[key] = value

    def __contains__(self, key: Union[Action, Observation]):
        return key in self.children


class QNode(TreeNode):
    def __init__(self, num_visits: int, 
                       value: float):
        super().__init__()
        self.num_visits = num_visits
        self.value = value
        self.children: Dict[Observation, "VNode"]  # o -> VNode
    
    def __getitem__(self, key: Observation) -> "VNode":
        item: VNode = super().__getitem__(key)
        return item

    def __str__(self):
        return "QNode(%d, %.3f | %s)" % (self.num_visits, self.value, str(self.children.keys()))

    def __repr__(self):
        return self.__str__()


class VNode(TreeNode):
    def __init__(self, num_visits: int, 
                       value: float):
        super().__init__()
        self.num_visits = num_visits
        self.value = value
        self.children: Dict[Action, "QNode"]       # a -> QNode

    def __getitem__(self, key: Action) -> "QNode":
        item: QNode = super().__getitem__(key)
        return item

    def __str__(self):
        return "VNode(%d, %.3f | %s)" % (self.num_visits, self.value, str(self.children.keys()))

    def __repr__(self):
        return self.__str__()

    def print_children_value(self):
        for action in self.children:
            print("   action %s: %.3f" % (str(action), self.children[action].value))
    
    def argmax(self):
        """
        Returns the action of the child with highest value
        """
        best_value = float("-inf")
        best_action = None
        for action in self.children:
            if self.children[action].value > best_value:
                best_action = action
                best_value = self.children[action].value  
        return best_action
    
    def sample(self):
        """
        Returns an action w.r.t #visit
        """
        actions = list(self.children.keys())
        p_actions = []
        for action in actions:
            p_actions.append(self.children[action].num_visits / self.num_visits)
        return random.choices(actions, p_actions)

        
class RootVNode(VNode):
    def __init__(self, num_visits: int, 
                       value: float, 
                       history: Tuple[HistoryEntry]):
        super().__init__(num_visits, value)
        self.history = history
    
    @classmethod
    def from_vnode(cls, vnode: VNode, 
                        history: Tuple[HistoryEntry]):
        """from_vnode(cls, vnode, history)"""
        rootnode = RootVNode(vnode.num_visits, vnode.value, history)
        rootnode.children = vnode.children
        return rootnode


class VNodeParticles(VNode):
    """POMCP-like planner's VNode maintains particle belief"""
    def __init__(self, num_visits: int, 
                       value: float, 
                       belief: WeightedParticles = WeightedParticles({})):
        super().__init__(num_visits, value)
        self.belief = belief
    
    def __str__(self):
        return "VNode(%d, %.3f, %d | %s)" % (self.num_visits, self.value, len(self.belief), str(self.children.keys()))
    
    def __repr__(self):
        return self.__str__()


class RootVNodeParticles(RootVNode):
    def __init__(self, num_visits: int, 
                       value: float, 
                       history: Tuple[HistoryEntry], 
                       belief: WeightedParticles = WeightedParticles({})):
        super().__init__(num_visits, value, history)
        self.belief = belief

    @classmethod
    def from_vnode(cls, vnode: VNodeParticles, 
                        history: Tuple[HistoryEntry]):
        rootnode = RootVNodeParticles(vnode.num_visits, vnode.value, history, belief=vnode.belief)
        rootnode.children = vnode.children
        return rootnode


class Planner:
    """
    A Planner can `plan` the next action to take for a given POMDP problem (online planning).
    The planner can be updated (which may also update the agent belief) once an action is executed and observation received.
    """

    def plan(self, **kwargs):    
        """
        The agent carries the information: Belief, history, BlackboxModel, and policy, necessary for planning
        Return:
            next_action: Action, *rest
        """
        raise NotImplementedError

    def update(self, real_action: Action, real_observation: Observation, reward: float):
        """
        Updates the planner based on real action and observation.
        Updates the agent accordingly if necessary.
        Args:
            real_action: Action
            real_observation: Observation
        """
        raise NotImplementedError