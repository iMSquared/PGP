import time
import math
import random
from copy import deepcopy
from typing import Tuple, Union, List


from pomdp.POMDP_framework import *
from pomdp.online_planner_framework import *

# Debugging
from debug.debug import debug_data

# Collecting data
from data_generation.collect_data import collect_trajectory_data


class POMCPOW(Planner):

    def __init__(self, pomdp: POMDP, config: Dict):
        """
        Args:
            max_depth (int): Depth of the MCTS tree. Default: 5.
            planning_time (float): amount of time given to each planning step (seconds). Defaults to -1.
                if negative, then planning terminates when number of simulations `num_sims` reached.
                If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
            num_sims (int): Number of simulations for each planning step.
                If negative, then will terminate when planning_time is reached.
                If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
            rollout_policy (RolloutPolicy): rollout policy. Default: RandomRollout.
            action_prior (ActionPrior|TODO|): a prior over preferred actions given state and history.
        """
        self.config = config
        self._pomdp = pomdp
        # To simplify function calls; plan only for one agent at a time
        self._agent = pomdp.agent

        # Configuration
        self._COLLECT_DATA    : bool = config["project_params"]["overridable"]["collect_data"]
        self._GUIDE_PREFERENCE: bool = config["project_params"]["overridable"]["guide_preference"]
        self._MAX_DEPTH       : int   = config["plan_params"]["max_depth"]
        self._MAX_DEPTH_REWARD: float = 0.
        self._NUM_SIMS        : int   = config["plan_params"]["num_sims"]
        self._PLANNING_TIME   : float = config["plan_params"]["planning_time"]
        if self._NUM_SIMS < 0 and self._PLANNING_TIME < 0:
            self._PLANNING_TIME = 1.
        self._NUM_VISITS_INIT   : int   = 0
        self._VALUE_INIT        : int   = 0.
        self._DISCOUNT_FACTOR   : float = config["plan_params"]["discount_factor"]
        self._EXPLORATION_CONST : float = config["plan_params"]["exploration_const"]["preference"] if self._GUIDE_PREFERENCE \
                                     else config["plan_params"]["exploration_const"]["value"]
        self._K_A               : float = config["plan_params"]["k_a"]
        self._ALPHA_A           : float = config["plan_params"]["alpha_a"]
        self._K_O               : float = config["plan_params"]["k_o"]
        self._ALPHA_O           : float = config["plan_params"]["alpha_o"]
        self._SELECT_BEST_ACTION: bool  = config["plan_params"]["select_best_action"]
        

        # For collecting data
        self.num_sim_total   = 0
        self.num_sim_success = 0
        self.sim_trajs       = []   # Only active when in COLLECT_DATA = True

    
    def plan(self) -> Tuple[Action, float, int, 
                            int, int,
                            List[Dict]]:
        """Plan!

        Returns:
            next_action (Action): Planned next action.
            time_taken (float): Time taken for planning one action.
            num_sim_total (int): Number of tried simulations.   
            num_sim_success (int): Number of simulation success. 
            sim_trajs (List[Dict]): Data of all trajectories. Activated when _COLLECT_DATA=True
        """

        # Validation
        #   Only works if the agent's belief is weighted particles
        if not isinstance(self._agent.belief, UnweightedParticles):
            raise TypeError("""Agent's belief is not represented in unweighted particles.
                            POMCPOW not usable. Please convert it to unweighted particles.""")


        # Initialize the root node of the search tree to the agent.
        # |TODO(ssh)|: This cause the root node to have unweighted particle.
        #   When looking into the POMCPOW paper, the root has unweighted particle....????
        if not hasattr(self._agent, "tree"):
            root = self._VNode(agent=self._agent, root=True)
            self._agent.add_attr("tree", root)


        # Logging variables...
        time_taken = 0.0
        start_time = time.time()
        self.num_sim_total = 0
        self.num_sim_success = 0

        # Data collection
        if self._COLLECT_DATA:
              self.sim_trajs = []

        # Planning loop
        for state in self._agent.belief:

            # Setting up the simulation of a particle
            #   Imagine simulation with sampled state
            self._agent.imagine_state(state, reset=True)
            #   Initialize simulation history as excuted history
            history = self._agent.history

            # Simulate a particle.
            total_reward = self._simulate(
                state   = state,
                history = history,
                root    = self._agent.tree,
                depth   = len(history) )
            
            # Update root V (V Backup)
            self._agent.tree.value = self._agent.tree.value\
                                   + (total_reward - self._agent.tree.value) / (self._agent.tree.num_visits)
            
            # Log cumulative time
            self.num_sim_total +=1
            time_taken = time.time() - start_time
            print("#sim:", self.num_sim_total, "accumulated_time:", time_taken)
            
        # Selecting the next action. 
        #   (next_action = best_action | sampling w.r.t the number of visits)
        if self._SELECT_BEST_ACTION:
            next_action = self._agent.tree.argmax()
        else:
            next_action = self._agent.tree.sample()


        return next_action, \
            time_taken, self.num_sim_total, self.num_sim_success, \
            self.sim_trajs
    


    def _simulate(self, state: State, 
                        history: Tuple[HistoryEntry], 
                        root: RootVNodeParticles, 
                        depth: int) -> float:
        """Implementation of `simulate(s, h, d)`
        root<-class:VNode, parent<-class:QNode
        
        Args:
            state (State): A state to simulate from
            history (Tuple[HistoryEntry]): History of the state particle
            root (RootVNodeParticles): Root node of the subtree to search.
            depth (int): Current depth

        Returns:
            float: The total discounted reward during the plannnig
        """

        # Max depth handling
        if depth >= self._MAX_DEPTH:
            # hao visit count +1
            root.num_visits += 1
            # Data collection (Max depth)
            if self._COLLECT_DATA:
                self.sim_trajs.append(
                    collect_trajectory_data(agent       = self._agent,
                                            history     = history,
                                            termination = TERMINATION_CONTINUE))
            # 0 reward for max depth
            return self._MAX_DEPTH_REWARD
        
        # Line 4: Select the best action child of the current state.
        action = self._action_progressive_widening(
            vnode   = root,
            history = history,
            state   = state)
        
        # Line 5: Generative model
        next_state, observation, reward, \
            termination = self._sample_generative_model(state, action)

        # Line 6~9: Observation resampling
        if len(root[action].children) <= self._K_O * root[action].num_visits ** self._ALPHA_O:
            if root[action][observation] is None:
                # |FIXME(Jiyong)|: M(hao) <- M(hao) + 1: 
                # It is okay to set every M as 1 for continuous observation, 
                # but not for discrete observation
                pass
        else:
            # |FIXME(Jiyong)|: o <- selet w.p. M(hao)/SIGMA_o M(hao)
            observation = random.choice(list(root[action].children.keys()))

            # # NOTE(ssh): Modified POMCPOW
            # #   o <- select w.p. Z(o|s,a,s') / SIMGA_o Z(o|s,a,s')
            # #   Lambda funcs
            # obs_to_weight_fn = lambda obs: \
            #     self._agent.get_weight(obs, next_state, action)
            # normalize_fn = lambda list_rel_likelihoods: \
            #     [rp/sum(list_rel_likelihoods) for rp in list_rel_likelihoods] # May suffer from division by 0.
            # #   Z(o|s,a,s') / SIMGA_o Z(o|s,a,s')
            # obs_children         = list(root[action].children.keys())
            # list_rel_likelihoods = list(map(obs_to_weight_fn, obs_children))
            # # NOTE(ssh): Resampling is only available the sum is not zero. Otherwise, just add new obs child.
            # if sum(list_rel_likelihoods) != 0:
            #     observation = random.choices(obs_children, weights=normalize_fn(list_rel_likelihoods))[0]


        # Line 11: Observation probability
        weight = self._agent.get_weight(observation, next_state, action)


        # Line 12~14: New observation node
        if observation not in root[action].children:

            # Line 10~11: append s` to B(hao) and Z(o|s,a,s`) to W(hao)
            root[action][observation] = self._VNode(agent=self._agent, root=False)
            root[action][observation].belief[next_state] = weight

            # Log history
            history += (HistoryEntry(action, observation, reward, PHASE_SIMULATION),)

            # Line 14: Rollout
            #   (a) Check termination before the rollout
            if (termination == TERMINATION_SUCCESS) or (termination == TERMINATION_FAIL):

                # Line 19~21: Update visit counts and Q values.
                if self._agent._value_model is not None:
                    if self._agent._value_model.is_v_model:
                        total_reward = reward + self._agent.get_value(history, next_state, self._agent.goal_condition)
                    else:
                        total_reward = self._agent.get_value(history, next_state, self._agent.goal_condition)
                else:
                    total_reward = reward
                root.num_visits += 1
                root[action].num_visits += 1
                root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)
                
                # Logging
                if termination == TERMINATION_SUCCESS:
                    self.num_sim_success += 1
                # Data collection
                if self._COLLECT_DATA:
                    self.sim_trajs.append(
                        collect_trajectory_data(agent       = self._agent,
                                                history     = history, 
                                                termination = termination))
                # Return with the reward when simulation terminates.
                return total_reward
            

            # Visit count += 1 when not a terminal node.
            root[action][observation].num_visits += 1
            #   (b) Replace rollout with the learned value
            if self._agent._value_model is not None:
                # V or Q
                if self._agent._value_model.is_v_model:
                    # Q(ha) = E[ R(h,a) + E_o [V'(h,a,o)] ]
                    accumulated_reward = self._agent.get_value(history, next_state, self._agent.goal_condition)
                    total_reward = reward + accumulated_reward
                else:
                    # Q(ha) = E[ Q'(h,a) ]
                    accumulated_reward = self._agent.get_value(history, next_state, self._agent.goal_condition)
                    total_reward = accumulated_reward
            #   (c) Rollout
            else:
                total_reward = reward + self._rollout(
                    state   = next_state,
                    history = history,
                    root    = root[action][observation],
                    depth   = depth + 1 )


        # Line 15~18: Existing (resampled) observation node
        else:

            # Line 10~11: Append s` to B(hao) and Z(o|s,a,s`) to W(hao)
            if next_state in root[action][observation].belief:
                root[action][observation].belief[next_state] += weight  # Same particle can be sampled twice.
            else:
                root[action][observation].belief[next_state] = weight

            # Line 16: s` <- select B(hao)[i] w.p W(hao)[i]/sigma(j=1~m) W(hao)[j]
            next_state = root[action][observation].belief.sample()
            #   Imagine simulation with resampled state
            self._agent.imagine_state(next_state, reset=False)

            # Line 17: r <- R(s,a,s`), Reset to the reward and the termination condition of the resampled next_state
            reward, termination = self._agent.blackbox_model.reward_model.sample(next_state, action, state)
            history += (HistoryEntry(action, observation, reward, PHASE_SIMULATION),)

            # Line 18: Recursive simulation
            #   (a) Check goal condition while simulating 
            if (termination == TERMINATION_SUCCESS) or (termination == TERMINATION_FAIL):

                # Line 19~21: Update visit counts and Q values.
                if self._agent._value_model is not None:
                    if self._agent._value_model.is_v_model:
                        total_reward = reward + self._agent.get_value(history, next_state, self._agent.goal_condition)
                    else:
                        total_reward = self._agent.get_value(history, next_state, self._agent.goal_condition)
                else:
                    total_reward = reward
                root.num_visits += 1
                root[action].num_visits += 1
                root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)

                # Logging
                if termination == TERMINATION_SUCCESS:
                    self.num_sim_success += 1
                # Data generation (Reached maximum depth)
                if self._COLLECT_DATA:
                    self.sim_trajs.append(
                        collect_trajectory_data(agent       = self._agent,
                                                history     = history, 
                                                termination = termination))
                # Return with the reward when simulation terminates.
                return total_reward
            
            #   (b) Proceed with recursion
            total_reward = reward + self._DISCOUNT_FACTOR * self._simulate(
                state   = next_state,
                history = history,
                root    = root[action][observation],
                depth   = depth + 1)


        # Line 19~21: Update visit counts and Q values.
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)
        # Backup V values (not included in actual POMCPOW)
        root[action][observation].value = root[action][observation].value\
                                        + (total_reward - root[action][observation].value) / (root[action][observation].num_visits)

        return total_reward



    def _rollout(self, state: State, history: Tuple[HistoryEntry], root: VNode, depth: int) -> float:
        """Implementation of POMCP Rollout
        This function does not manipulate any tree nodes in the planner, 
        i.e. VNode or QNode. It only uses the POMDP model.

        Args:
            state (State): The initial state to start the rollout from.
            history (Tuple[HistoryEntry]): The history of the state
            depth (int): The depth of the state.

        Returns:
            float: total_discounted_reward
        """
        discount = self._DISCOUNT_FACTOR
        total_discounted_reward = 0.0

        while depth < self._MAX_DEPTH:
            action = self._agent.get_action(history, state, goal=self._agent.goal_condition, rollout=True)
            
            next_state, observation, reward, termination = self._sample_generative_model(state, action)
            history += (HistoryEntry(action, observation, reward, PHASE_ROLLOUT),)

            depth += 1
            total_discounted_reward += reward * discount
            discount *= self._DISCOUNT_FACTOR
            state = next_state

            # Check goal condition while rollout
            if (termination == TERMINATION_SUCCESS) or (termination == TERMINATION_FAIL):
                # Logging
                if termination == TERMINATION_SUCCESS:
                    self.num_sim_success += 1
                # Data collection (rollout reached termination condition)
                if self._COLLECT_DATA:
                      self.sim_trajs.append(
                        collect_trajectory_data(agent       = self._agent, 
                                                history     = history,
                                                termination = termination))
                # Return reward
                return total_discounted_reward

        # Data collection (rollout reached max depth)
        if self._COLLECT_DATA:
              self.sim_trajs.append(
                collect_trajectory_data(agent       = self._agent,
                                        history     = history,
                                        termination = TERMINATION_CONTINUE))
              
        # Max depth penalty
        total_discounted_reward += self._MAX_DEPTH_REWARD * discount
        # Return reward
        return total_discounted_reward



    def _action_progressive_widening(self, vnode: RootVNode, history: Tuple[HistoryEntry], state: State) -> Action:
        """Implementation of action progressive widening.
        1. Appends a new action child if the number of action child 
           is below the (k_a * vnode.num_visits ** alpha_a).
        2. Returns the next action child, where a = argmax_a UCB(s, a).
           
        Args:
            vnode (VNode): Vnode to evaluate.
            history (Tuple[HistoryEntry]): The history will be passed to the policy models.
            state (State): The particle state will be passed to the policy model.

        Returns:
            Action: Selected action.
        """
        
        if len(vnode.children) <= self._K_A * vnode.num_visits ** self._ALPHA_A:
            _action = self._agent.get_action(history, state, goal=self._agent.goal_condition, rollout=False)

            if vnode[_action] is None:
                history_action_node = QNode(self._NUM_VISITS_INIT, self._VALUE_INIT)
                vnode[_action] = history_action_node

        return self._ucb(vnode)



    def _ucb(self, root: VNode) -> Action:
        """UCB1
        Selects the action child of the `root` with the best UCB value.

        Args:
            root (VNode): A vnode to select the action child from.

        Returns:
            Action: Selected action.
        """
        best_action, best_value = None, float('-inf')
        for action in root.children:
            if root[action].num_visits == 0:
                val = float('inf')
            else:
                val = root[action].value + \
                    self._EXPLORATION_CONST * math.sqrt(math.log(root.num_visits + 1) / root[action].num_visits)
                    
            if val > best_value:
                best_action = action
                best_value = val

        return best_action



    def _sample_generative_model(self, state: State, action: Action) \
                                        -> Tuple[ State, Observation, float, str ]:
        """(s', o, r) ~ G(s, a)

        Args:
            state (State): Current state
            action (Action): Action to do.

        Returns:
            next_state (State): Generated next state
            observation (Observation): Generated observation 
            reward (float): Generated reward
            termination (str): Termination condition. "success" or "fail" or None.
        """
        
        next_state, observation, reward, termination = self._agent.blackbox_model.sample(state, action)
        
        return next_state, observation, reward, termination



    def _VNode(self, agent: Agent=None, root: bool=False) -> Union[ RootVNodeParticles, VNodeParticles ]:
        """
        Returns a VNode with default values;
        The function naming makes it clear that this function is about creating a VNode object.
        """
        if root:
            # agent cannot be None.
            return RootVNodeParticles(self._NUM_VISITS_INIT,
                                      self._VALUE_INIT,
                                      agent.history,
                                      belief=deepcopy(agent.belief))
        else:
            return VNodeParticles(self._NUM_VISITS_INIT,
                                    self._VALUE_INIT,
                                    belief=WeightedParticles({}))



    # |FIXME(Jiyong)|: how to implement changing root node to existing node in discrete case and particle reinvigoration.
    def update(self, agent: Agent, real_action: Action, real_observation: Observation):
        agent.tree[real_action][real_observation] = self._VNode(agent=agent, root=True)
        agent.tree = RootVNodeParticles.from_vnode(agent.tree[real_action][real_observation], agent.history)