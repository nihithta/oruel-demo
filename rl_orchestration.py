"""
Datacenter Power Orchestration System with Reinforcement Learning
Based on: "Reinforcement learning for data center energy efficiency optimization"

This system integrates with physics-based predictive maintenance to provide
agentic power, cooling, and compute orchestration when anomalies are detected.

Key Components:
1. State Space: GPU telemetry, Power BMS, HVAC data, failure predictions
2. Action Space: Power adjustments, cooling controls, workload migration
3. Reward Function: Energy efficiency + reliability + QoS maintenance
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Failure types from predictive maintenance system"""
    POWER_SUPPLY = "power_supply"
    THERMAL_OVERLOAD = "thermal_overload"
    GPU_DEGRADATION = "gpu_degradation"
    COOLING_SYSTEM = "cooling_system"
    BATTERY_DEGRADATION = "battery_degradation"


class ActionType(Enum):
    """Available orchestration actions"""
    # Power Actions
    REDUCE_POWER_LIMIT = "reduce_power_limit"
    INCREASE_POWER_LIMIT = "increase_power_limit"
    SWITCH_POWER_SOURCE = "switch_power_source"
    ENABLE_BATTERY_BACKUP = "enable_battery_backup"
    
    # Cooling Actions
    INCREASE_COOLING = "increase_cooling"
    DECREASE_COOLING = "decrease_cooling"
    ADJUST_CRAC_SETPOINT = "adjust_crac_setpoint"
    ENABLE_FREE_COOLING = "enable_free_cooling"
    
    # Compute Actions
    MIGRATE_WORKLOAD = "migrate_workload"
    REDUCE_GPU_FREQUENCY = "reduce_gpu_frequency"
    THROTTLE_COMPUTE = "throttle_compute"
    REDISTRIBUTE_LOAD = "redistribute_load"
    
    # Emergency Actions
    GRACEFUL_SHUTDOWN = "graceful_shutdown"
    FAILOVER_SWITCH = "failover_switch"


@dataclass
class MaintenancePrediction:
    """Output from physics-based predictive maintenance system"""
    failure_probability: float  # 0-1
    remaining_useful_life: float  # hours
    failure_type: FailureType
    root_cause: str
    degradation_trajectory: List[float]
    causal_graph: Dict[str, Any]
    confidence: float


@dataclass
class DatacenterState:
    """Current state of the datacenter"""
    # GPU Telemetry
    gpu_utilization: List[float]
    gpu_temperature: List[float]
    gpu_power_draw: List[float]
    gpu_memory_usage: List[float]
    
    # Power BMS Telemetry
    total_power_consumption: float
    power_per_rack: List[float]
    voltage_levels: List[float]
    current_levels: List[float]
    battery_soc: float  # State of charge
    grid_power_available: bool
    
    # HVAC Data
    rack_inlet_temp: List[float]
    rack_outlet_temp: List[float]
    crac_supply_temp: float
    crac_return_temp: float
    humidity: float
    airflow_rate: float
    
    # Predictive Maintenance Outputs
    maintenance_prediction: Optional[MaintenancePrediction]
    
    # Performance Metrics
    pue: float  # Power Usage Effectiveness
    workload_completion_rate: float
    sla_violations: int
    
    timestamp: datetime


@dataclass
class OrchestrationAction:
    """Orchestration action to be taken"""
    action_type: ActionType
    target_component: str
    parameters: Dict[str, Any]
    priority: int  # 1-10, 10 being highest
    estimated_impact: Dict[str, float]
    rollback_possible: bool


class PowerOrchestrationAgent:
    """
    Deep Reinforcement Learning Agent for Power Orchestration
    Based on SAC (Soft Actor-Critic) algorithm - most common in the paper
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # State space dimensions
        self.n_gpus = config.get('n_gpus', 8)
        self.n_racks = config.get('n_racks', 10)
        
        # Action space boundaries
        self.power_limit_range = (150, 400)  # Watts per GPU
        self.cooling_setpoint_range = (18, 27)  # Celsius
        self.frequency_scale_range = (0.5, 1.0)  # GPU frequency multiplier
        
        # Safety thresholds
        self.max_gpu_temp = config.get('max_gpu_temp', 85)
        self.max_rack_temp = config.get('max_rack_temp', 35)
        self.min_battery_soc = config.get('min_battery_soc', 20)
        
        # RL hyperparameters (from paper recommendations)
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 3e-4
        self.tau = 0.005  # Soft update coefficient
        
        # Initialize Q-values table (simplified - in production use neural networks)
        self.initialize_value_functions()
        
        logger.info("Power Orchestration Agent initialized")
    
    def initialize_value_functions(self):
        """Initialize value functions for RL agent"""
        # In production, these would be neural networks (as per paper)
        # Here we use simplified representation
        self.state_value = {}
        self.action_value = {}
        self.policy = {}
    
    def encode_state(self, state: DatacenterState) -> np.ndarray:
        """
        Encode datacenter state into feature vector
        Based on MDP state formulation from the paper
        """
        features = []
        
        # GPU features (normalized)
        features.extend([
            np.mean(state.gpu_utilization) / 100,
            np.max(state.gpu_temperature) / 100,
            np.mean(state.gpu_power_draw) / 400,
            np.std(state.gpu_utilization) / 100
        ])
        
        # Power features
        features.extend([
            state.total_power_consumption / (self.n_racks * 10000),  # Normalize by max capacity
            state.battery_soc / 100,
            float(state.grid_power_available)
        ])
        
        # Thermal features
        features.extend([
            np.mean(state.rack_inlet_temp) / 50,
            np.max(state.rack_outlet_temp) / 50,
            state.crac_supply_temp / 50,
            (state.crac_return_temp - state.crac_supply_temp) / 20  # Temperature delta
        ])
        
        # Predictive maintenance features
        if state.maintenance_prediction:
            pred = state.maintenance_prediction
            features.extend([
                pred.failure_probability,
                min(pred.remaining_useful_life / 720, 1.0),  # Normalize to 30 days
                pred.confidence
            ])
        else:
            features.extend([0, 1.0, 1.0])  # No prediction = no failure
        
        # Performance features
        features.extend([
            state.pue / 2.0,  # Normalize (typical range 1.0-2.0)
            state.workload_completion_rate,
            min(state.sla_violations / 10, 1.0)
        ])
        
        return np.array(features)
    
    def compute_reward(self, 
                      prev_state: DatacenterState, 
                      action: OrchestrationAction,
                      new_state: DatacenterState) -> float:
        """
        Compute reward function (multi-objective as per paper)
        Balances: Energy efficiency + Reliability + QoS
        """
        reward = 0.0
        
        # 1. Energy Efficiency Reward (40% weight)
        pue_improvement = (prev_state.pue - new_state.pue) / prev_state.pue
        energy_reduction = (prev_state.total_power_consumption - 
                          new_state.total_power_consumption) / prev_state.total_power_consumption
        
        energy_reward = 100 * (0.5 * pue_improvement + 0.5 * energy_reduction)
        reward += 0.4 * energy_reward
        
        # 2. Reliability Reward (40% weight)
        reliability_reward = 0.0
        
        if new_state.maintenance_prediction:
            pred = new_state.maintenance_prediction
            # Penalize high failure probability
            reliability_reward -= 50 * pred.failure_probability
            
            # Reward increasing RUL
            if prev_state.maintenance_prediction:
                rul_improvement = (new_state.maintenance_prediction.remaining_useful_life - 
                                 prev_state.maintenance_prediction.remaining_useful_life)
                reliability_reward += 10 * np.clip(rul_improvement / 24, -1, 1)
        
        # Penalize thermal violations
        max_temp = np.max(new_state.gpu_temperature)
        if max_temp > self.max_gpu_temp:
            reliability_reward -= 100 * (max_temp - self.max_gpu_temp)
        
        reward += 0.4 * reliability_reward
        
        # 3. QoS Reward (20% weight)
        qos_reward = 0.0
        
        # Reward workload completion
        qos_reward += 50 * new_state.workload_completion_rate
        
        # Penalize SLA violations
        sla_penalty = (new_state.sla_violations - prev_state.sla_violations)
        qos_reward -= 20 * sla_penalty
        
        reward += 0.2 * qos_reward
        
        return reward
    
    def select_action(self, state: DatacenterState, mode='exploit') -> OrchestrationAction:
        """
        Select orchestration action based on current state
        Uses epsilon-greedy for exploration (as per paper recommendations)
        """
        state_vector = self.encode_state(state)
        
        # Check if emergency action needed
        if self.requires_emergency_action(state):
            return self.get_emergency_action(state)
        
        # Get available actions based on state
        available_actions = self.get_available_actions(state)
        
        if mode == 'explore' and np.random.random() < 0.1:  # 10% exploration
            # Random action
            action_idx = np.random.randint(len(available_actions))
            return available_actions[action_idx]
        else:
            # Greedy action (highest Q-value)
            best_action = self.get_best_action(state, available_actions)
            return best_action
    
    def requires_emergency_action(self, state: DatacenterState) -> bool:
        """Check if emergency action is required"""
        # Critical temperature
        if np.max(state.gpu_temperature) > self.max_gpu_temp + 5:
            return True
        
        # Critical failure probability
        if state.maintenance_prediction and state.maintenance_prediction.failure_probability > 0.9:
            return True
        
        # Critical battery level with no grid
        if not state.grid_power_available and state.battery_soc < self.min_battery_soc:
            return True
        
        return False
    
    def get_emergency_action(self, state: DatacenterState) -> OrchestrationAction:
        """Generate emergency action"""
        # Determine emergency type
        if np.max(state.gpu_temperature) > self.max_gpu_temp + 5:
            return OrchestrationAction(
                action_type=ActionType.THROTTLE_COMPUTE,
                target_component="all_gpus",
                parameters={'throttle_level': 0.5, 'duration': 300},
                priority=10,
                estimated_impact={'temp_reduction': 10, 'performance_impact': -50},
                rollback_possible=True
            )
        
        if state.maintenance_prediction and state.maintenance_prediction.failure_probability > 0.9:
            if state.maintenance_prediction.failure_type == FailureType.POWER_SUPPLY:
                return OrchestrationAction(
                    action_type=ActionType.FAILOVER_SWITCH,
                    target_component=state.maintenance_prediction.root_cause,
                    parameters={'backup_source': 'battery'},
                    priority=10,
                    estimated_impact={'downtime': 0},
                    rollback_possible=False
                )
        
        return OrchestrationAction(
            action_type=ActionType.GRACEFUL_SHUTDOWN,
            target_component="critical_systems",
            parameters={'shutdown_order': ['non_critical', 'batch_jobs', 'critical']},
            priority=10,
            estimated_impact={'service_impact': -100},
            rollback_possible=False
        )
    
    def get_available_actions(self, state: DatacenterState) -> List[OrchestrationAction]:
        """
        Generate list of available actions based on current state
        Implements the action space from the paper
        """
        actions = []
        
        # Predictive maintenance guided actions
        if state.maintenance_prediction:
            pred = state.maintenance_prediction
            
            if pred.failure_type == FailureType.THERMAL_OVERLOAD:
                # Cooling-related actions
                actions.append(OrchestrationAction(
                    action_type=ActionType.INCREASE_COOLING,
                    target_component="crac_units",
                    parameters={'increment': 5, 'airflow_increase': 10},
                    priority=8,
                    estimated_impact={'temp_reduction': 3, 'power_increase': 5},
                    rollback_possible=True
                ))
                
                actions.append(OrchestrationAction(
                    action_type=ActionType.REDISTRIBUTE_LOAD,
                    target_component="hot_racks",
                    parameters={'target_racks': self.identify_cool_racks(state)},
                    priority=7,
                    estimated_impact={'temp_reduction': 5, 'migration_time': 120},
                    rollback_possible=True
                ))
            
            elif pred.failure_type == FailureType.POWER_SUPPLY:
                # Power-related actions
                actions.append(OrchestrationAction(
                    action_type=ActionType.REDUCE_POWER_LIMIT,
                    target_component="gpu_cluster",
                    parameters={'new_limit': 300},
                    priority=8,
                    estimated_impact={'power_reduction': 15, 'performance_impact': -5},
                    rollback_possible=True
                ))
                
                actions.append(OrchestrationAction(
                    action_type=ActionType.ENABLE_BATTERY_BACKUP,
                    target_component="critical_loads",
                    parameters={'duration': 3600},
                    priority=9,
                    estimated_impact={'reliability_increase': 20},
                    rollback_possible=True
                ))
            
            elif pred.failure_type == FailureType.GPU_DEGRADATION:
                # Compute workload actions
                actions.append(OrchestrationAction(
                    action_type=ActionType.MIGRATE_WORKLOAD,
                    target_component=pred.root_cause,
                    parameters={'destination': 'healthy_gpus'},
                    priority=7,
                    estimated_impact={'performance_maintained': 95},
                    rollback_possible=True
                ))
        
        # General optimization actions (always available)
        if state.pue > 1.5:
            # PUE optimization
            actions.append(OrchestrationAction(
                action_type=ActionType.ADJUST_CRAC_SETPOINT,
                target_component="crac_units",
                parameters={'new_setpoint': min(state.crac_supply_temp + 1, 27)},
                priority=5,
                estimated_impact={'pue_improvement': 0.02, 'energy_saving': 2},
                rollback_possible=True
            ))
        
        # Free cooling opportunity
        if np.mean(state.rack_inlet_temp) < 20 and state.humidity < 60:
            actions.append(OrchestrationAction(
                action_type=ActionType.ENABLE_FREE_COOLING,
                target_component="cooling_system",
                parameters={'mode': 'economizer'},
                priority=6,
                estimated_impact={'cooling_energy_reduction': 30},
                rollback_possible=True
            ))
        
        return actions
    
    def get_best_action(self, state: DatacenterState, 
                       available_actions: List[OrchestrationAction]) -> OrchestrationAction:
        """
        Select best action using Q-value estimation
        In production: use neural network for Q-value approximation
        """
        if not available_actions:
            # No action
            return OrchestrationAction(
                action_type=ActionType.REDUCE_POWER_LIMIT,  # Safe default
                target_component="none",
                parameters={},
                priority=0,
                estimated_impact={},
                rollback_possible=True
            )
        
        # Simplified Q-value estimation based on estimated impacts
        best_action = None
        best_score = -float('inf')
        
        for action in available_actions:
            # Score based on estimated impact and priority
            score = action.priority * 10
            
            for metric, value in action.estimated_impact.items():
                if 'reduction' in metric or 'saving' in metric or 'improvement' in metric:
                    score += value
                elif 'increase' in metric and 'power' not in metric:
                    score += value
                else:
                    score -= abs(value) * 0.1
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def identify_cool_racks(self, state: DatacenterState) -> List[int]:
        """Identify racks with lower temperatures for load redistribution"""
        temps = np.array(state.rack_outlet_temp)
        median_temp = np.median(temps)
        cool_racks = [i for i, temp in enumerate(temps) if temp < median_temp]
        return cool_racks
    
    def update_policy(self, experience: Tuple):
        """
        Update policy based on experience (state, action, reward, next_state)
        Implements SAC update rule (simplified)
        """
        state, action, reward, next_state, done = experience
        
        # In production: implement full SAC update with neural networks
        # This is a placeholder for the update logic
        state_key = self.state_to_key(state)
        
        # Store experience for learning
        if state_key not in self.state_value:
            self.state_value[state_key] = 0
        
        # Temporal difference update (simplified)
        if not done:
            next_state_key = self.state_to_key(next_state)
            next_value = self.state_value.get(next_state_key, 0)
            td_target = reward + self.gamma * next_value
        else:
            td_target = reward
        
        # Update value
        self.state_value[state_key] += self.learning_rate * (td_target - self.state_value[state_key])
    
    def state_to_key(self, state: DatacenterState) -> str:
        """Convert state to hashable key for value function"""
        state_vector = self.encode_state(state)
        # Discretize for table lookup
        discretized = tuple(np.round(state_vector, decimals=1))
        return str(discretized)


class PowerOrchestrationCoordinator:
    """
    Main coordinator that integrates with predictive maintenance
    and executes orchestration actions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent = PowerOrchestrationAgent(config)
        self.action_history = []
        self.state_history = []
        
        logger.info("Power Orchestration Coordinator initialized")
    
    def process_anomaly(self, 
                       current_state: DatacenterState,
                       maintenance_prediction: MaintenancePrediction) -> List[OrchestrationAction]:
        """
        Main entry point when anomaly is detected by predictive maintenance
        
        Returns: List of orchestration actions to execute
        """
        logger.info(f"Processing anomaly: {maintenance_prediction.failure_type.value}")
        logger.info(f"Failure probability: {maintenance_prediction.failure_probability:.2%}")
        logger.info(f"Remaining Useful Life: {maintenance_prediction.remaining_useful_life:.1f} hours")
        
        # Update state with prediction
        current_state.maintenance_prediction = maintenance_prediction
        
        # Select action using RL agent
        action = self.agent.select_action(current_state, mode='exploit')
        
        # Validate action safety
        if self.validate_action_safety(action, current_state):
            # Store for learning
            self.state_history.append(current_state)
            self.action_history.append(action)
            
            logger.info(f"Selected action: {action.action_type.value}")
            logger.info(f"Target: {action.target_component}")
            logger.info(f"Priority: {action.priority}/10")
            
            return [action]
        else:
            logger.warning("Action failed safety validation, selecting alternative")
            return self.get_safe_fallback_action(current_state)
    
    def validate_action_safety(self, 
                              action: OrchestrationAction, 
                              state: DatacenterState) -> bool:
        """
        Validate that action won't cause safety violations
        Implements safety constraints from the paper
        """
        # Check thermal safety
        if action.action_type in [ActionType.REDUCE_POWER_LIMIT, ActionType.MIGRATE_WORKLOAD]:
            return True  # These are always safe
        
        # Check if cooling reduction is safe
        if action.action_type == ActionType.DECREASE_COOLING:
            if np.max(state.gpu_temperature) > self.agent.max_gpu_temp - 5:
                return False
        
        # Check power safety
        if action.action_type == ActionType.INCREASE_POWER_LIMIT:
            if state.total_power_consumption > self.config.get('max_power_capacity', 100000) * 0.9:
                return False
        
        return True
    
    def get_safe_fallback_action(self, state: DatacenterState) -> List[OrchestrationAction]:
        """Get conservative safe action"""
        return [OrchestrationAction(
            action_type=ActionType.REDUCE_POWER_LIMIT,
            target_component="all_systems",
            parameters={'reduction_percent': 10},
            priority=5,
            estimated_impact={'safety_margin': 15},
            rollback_possible=True
        )]
    
    def execute_action(self, action: OrchestrationAction) -> Dict[str, Any]:
        """
        Execute orchestration action
        In production: interfaces with actual datacenter control systems
        """
        execution_plan = {
            'action_id': len(self.action_history),
            'timestamp': datetime.now().isoformat(),
            'action_type': action.action_type.value,
            'target': action.target_component,
            'parameters': action.parameters,
            'estimated_completion_time': self.estimate_execution_time(action),
            'rollback_procedure': self.generate_rollback_procedure(action) if action.rollback_possible else None
        }
        
        logger.info(f"Executing action: {json.dumps(execution_plan, indent=2)}")
        
        # In production: send to datacenter management system
        # For now, return execution plan
        return execution_plan
    
    def estimate_execution_time(self, action: OrchestrationAction) -> int:
        """Estimate time to complete action (seconds)"""
        timing_map = {
            ActionType.REDUCE_POWER_LIMIT: 30,
            ActionType.INCREASE_COOLING: 120,
            ActionType.MIGRATE_WORKLOAD: 300,
            ActionType.REDISTRIBUTE_LOAD: 600,
            ActionType.ENABLE_FREE_COOLING: 180,
            ActionType.GRACEFUL_SHUTDOWN: 900
        }
        return timing_map.get(action.action_type, 60)
    
    def generate_rollback_procedure(self, action: OrchestrationAction) -> Dict[str, Any]:
        """Generate rollback procedure for action"""
        return {
            'revert_action': True,
            'restore_parameters': 'previous_state',
            'validation_required': True,
            'max_rollback_time': 300
        }
    
    def learn_from_execution(self, 
                           prev_state: DatacenterState,
                           action: OrchestrationAction,
                           new_state: DatacenterState):
        """
        Update RL agent based on execution results
        Implements online learning as recommended in the paper
        """
        reward = self.agent.compute_reward(prev_state, action, new_state)
        
        logger.info(f"Learning from execution. Reward: {reward:.2f}")
        
        # Update policy
        experience = (prev_state, action, reward, new_state, False)
        self.agent.update_policy(experience)


def main_demo():
    """Demonstration of the power orchestration system"""
    
    # Configuration
    config = {
        'n_gpus': 8,
        'n_racks': 10,
        'max_gpu_temp': 85,
        'max_rack_temp': 35,
        'min_battery_soc': 20,
        'max_power_capacity': 100000
    }
    
    # Initialize coordinator
    coordinator = PowerOrchestrationCoordinator(config)
    
    # Simulate anomaly detection from predictive maintenance
    maintenance_prediction = MaintenancePrediction(
        failure_probability=0.75,
        remaining_useful_life=12.5,  # hours
        failure_type=FailureType.THERMAL_OVERLOAD,
        root_cause="Rack-05 cooling degradation",
        degradation_trajectory=[0.1, 0.3, 0.5, 0.7, 0.75],
        causal_graph={'root': 'cooling_pump', 'affected': ['rack_05', 'rack_06']},
        confidence=0.89
    )
    
    # Current datacenter state
    current_state = DatacenterState(
        gpu_utilization=[85, 90, 78, 92, 88, 75, 82, 87],
        gpu_temperature=[78, 82, 76, 84, 80, 75, 79, 83],
        gpu_power_draw=[320, 340, 310, 350, 330, 305, 325, 335],
        gpu_memory_usage=[75, 80, 70, 85, 78, 72, 76, 81],
        total_power_consumption=85000,
        power_per_rack=[8500]*10,
        voltage_levels=[480]*10,
        current_levels=[17.7]*10,
        battery_soc=85,
        grid_power_available=True,
        rack_inlet_temp=[22, 23, 21, 28, 27, 22, 23, 24, 22, 23],  # Rack 5 elevated
        rack_outlet_temp=[32, 34, 31, 42, 40, 33, 34, 35, 32, 33],  # Rack 5 hot
        crac_supply_temp=20,
        crac_return_temp=35,
        humidity=45,
        airflow_rate=8500,
        maintenance_prediction=None,
        pue=1.58,
        workload_completion_rate=0.95,
        sla_violations=2,
        timestamp=datetime.now()
    )
    
    # Process anomaly
    actions = coordinator.process_anomaly(current_state, maintenance_prediction)
    
    # Execute actions
    for action in actions:
        execution_plan = coordinator.execute_action(action)
        print(f"\n{'='*60}")
        print("EXECUTION PLAN:")
        print(json.dumps(execution_plan, indent=2))
    
    # Simulate new state after action
    new_state = DatacenterState(
        gpu_utilization=[75, 80, 70, 82, 78, 70, 75, 77],  # Reduced after migration
        gpu_temperature=[75, 78, 73, 78, 76, 72, 75, 77],  # Cooler
        gpu_power_draw=[300, 320, 295, 310, 305, 290, 300, 310],
        gpu_memory_usage=[70, 75, 65, 80, 73, 68, 72, 76],
        total_power_consumption=82000,  # Reduced
        power_per_rack=[8200]*10,
        voltage_levels=[480]*10,
        current_levels=[17.1]*10,
        battery_soc=85,
        grid_power_available=True,
        rack_inlet_temp=[22, 23, 21, 24, 23, 22, 23, 24, 22, 23],  # Rack 5 improved
        rack_outlet_temp=[32, 34, 31, 36, 34, 33, 34, 35, 32, 33],  # Better
        crac_supply_temp=19,  # Adjusted
        crac_return_temp=34,
        humidity=45,
        airflow_rate=9000,  # Increased
        maintenance_prediction=maintenance_prediction,
        pue=1.52,  # Improved
        workload_completion_rate=0.96,  # Maintained
        sla_violations=2,
        timestamp=datetime.now()
    )
    
    # Learn from execution
    coordinator.learn_from_execution(current_state, actions[0], new_state)
    
    print(f"\n{'='*60}")
    print("ORCHESTRATION COMPLETE")
    print(f"PUE improved from {current_state.pue:.2f} to {new_state.pue:.2f}")
    print(f"Power reduced from {current_state.total_power_consumption}W to {new_state.total_power_consumption}W")
    print(f"Max GPU temp reduced from {max(current_state.gpu_temperature)}°C to {max(new_state.gpu_temperature)}°C")


if __name__ == "__main__":
    main_demo()