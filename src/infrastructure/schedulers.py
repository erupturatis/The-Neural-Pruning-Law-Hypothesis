from typing import Dict
import numpy as np

from src.infrastructure.configs_general import VERBOSE_SCHEDULER
from src.infrastructure.constants import SCHEDULER_MESSAGE
from src.infrastructure.others import prefix_path_with_root

populated_values = { 
    "epochs": [], 
    "remaining": [], 
    "saliency": [],
    "accuracy": []
}

def get_populated_values() -> list: 
    return populated_values

class PressureSchedulerPolicyNeuralPruningLaw:
    def __init__(self, pressure_exponent_constant: float, sparsity_target: float, epochs_target: int, step_size:float = 0.3, aggresivity: int = 6):
        self.gamma = 0

        self.step_size = step_size
        self.remaining_params_target = 100-sparsity_target
        self.epochs_target = epochs_target
        self.sparsity_target = sparsity_target
        self.recorded_states = []
        self.inertia_positive = 0
        self.inertia_negative = 0
        self.EXP = pressure_exponent_constant

        # Empirically found formulas, work well for any setup
        late_aggressivity = epochs_target / 2
        aggressivity_transition = aggresivity / epochs_target

        self.trajectory_calculator = TrajectoryCalculator(
            pruning_target=self.remaining_params_target,
            epochs_target=self.epochs_target,
            late_aggresivity=late_aggressivity,
            aggresivity_transition=aggressivity_transition
        )

    def _get_expected_sparsity(self, epoch:int) -> float:
        if epoch <= self.epochs_target:
            return self.trajectory_calculator.get_expected_pruning_at_epoch(epoch)

    def step(self, epoch: int, current_sparsity: float) -> None:
        if epoch >= self.epochs_target:
            return

        expected_sparsity = self._get_expected_sparsity(epoch)

        if VERBOSE_SCHEDULER:
            print(SCHEDULER_MESSAGE + "Current params", current_sparsity , "Expected params", expected_sparsity )

        if current_sparsity > expected_sparsity:
            # network has too many params, prune more aggresive
            populated_values["epochs"].append(epoch)
            populated_values["remaining"].append(current_sparsity)
            populated_values["saliency"].append(self.gamma)
            populated_values["accuracy"].append(self.acc)
            # expected deviation
            self.gamma += self.step_size + self.inertia_positive
            self.inertia_positive += self.step_size/4
            self.inertia_negative = 0
            if VERBOSE_SCHEDULER:
                print(SCHEDULER_MESSAGE + "SCHEDULER::Increasing pressure ", self.gamma)
        else:
            self.inertia_positive = 0
            if VERBOSE_SCHEDULER:
                print(SCHEDULER_MESSAGE + "SCHEDULER::Decreasing pressure ", self.gamma)

    def get_multiplier(self) -> int:
        return self.gamma ** self.EXP
    

class PressureSchedulerPolicy1:
    def __init__(self, pressure_exponent_constant: float, sparsity_target: float, epochs_target: int, step_size:float = 0.3, aggresivity: int = 6):
        self.gamma = 0

        self.step_size = step_size
        self.remaining_params_target = 100-sparsity_target
        self.epochs_target = epochs_target
        self.sparsity_target = sparsity_target
        self.recorded_states = []
        self.inertia_positive = 0
        self.inertia_negative = 0
        self.EXP = pressure_exponent_constant

        # Empirically found formulas, work well for any setup
        late_aggressivity = epochs_target / 2
        aggressivity_transition = aggresivity / epochs_target

        self.trajectory_calculator = TrajectoryCalculator(
            pruning_target=self.remaining_params_target,
            epochs_target=self.epochs_target,
            late_aggresivity=late_aggressivity,
            aggresivity_transition=aggressivity_transition
        )

    def _get_expected_sparsity(self, epoch:int) -> float:
        if epoch <= self.epochs_target:
            return self.trajectory_calculator.get_expected_pruning_at_epoch(epoch)

    def step(self, epoch: int, current_sparsity: float) -> None:
        if epoch >= self.epochs_target:
            return

        expected_sparsity = self._get_expected_sparsity(epoch)

        if VERBOSE_SCHEDULER:
            print(SCHEDULER_MESSAGE + "Current params", current_sparsity , "Expected params", expected_sparsity )

        if current_sparsity > expected_sparsity:
            # network has too many params, prune more aggresive
            # expected deviation
            self.gamma += self.step_size + self.inertia_positive
            self.inertia_positive += self.step_size/4
            self.inertia_negative = 0
            if VERBOSE_SCHEDULER:
                print(SCHEDULER_MESSAGE + "SCHEDULER::Increasing pressure ", self.gamma)
        else:
            # Ease up presssure
            self.gamma -= self.step_size + self.inertia_negative
            # self.gamma /= 2
            self.inertia_negative += self.step_size/4
            self.inertia_positive = 0

            if self.gamma < 0:
                self.gamma = 0

            if VERBOSE_SCHEDULER:
                print(SCHEDULER_MESSAGE + "SCHEDULER::Decreasing pressure ", self.gamma)

    def get_multiplier(self) -> int:
        return self.gamma ** self.EXP
    

class PressureSchedulerPolicyMeasurements:
    def __init__(self, exponent_start: int, exponent_end: int, base: float, epochs_raise: int = 50):
        self.gamma = 0
        self.exponent_start = exponent_start 
        self.exponent_end = exponent_end 
        self.epochs_raise = epochs_raise 
        self.base = base 
        self.gamma = self.base
        self.EXP = exponent_start


    def step(self, epoch: int, *args) -> None:
        print(epoch, self.epochs_raise)
        if not epoch % self.epochs_raise == 0:
            return

        self.EXP += 1

    def get_multiplier(self) -> int:
        return self.gamma ** self.EXP

import math
def cosine_function(epoch, period=60):
    return math.cos(2 * math.pi * (epoch % period) / period)

def expected_pruning_decrease_at_epoch(epoch, start_decrease, end_decrease, aggressivity_transition, late_aggressivity):
    """
    Calculates the expected pruning decrease at a given epoch using a sigmoid function.

    Parameters:
        epoch (int): Current epoch.
        start_decrease (float): Initial pruning decrease factor.
        end_decrease (float): Final pruning decrease factor.
        aggressivity_transition (float): Controls the steepness of the sigmoid transition.
        late_aggressivity (float): Epoch at which the pruning aggressiveness starts to increase.

    Returns:
        float: Pruning decrease factor at the given epoch.
    """
    sigmoid = 1 / (1 + np.exp(-aggressivity_transition * (epoch - late_aggressivity)))
    return start_decrease + (end_decrease - start_decrease) * sigmoid

def expected_pruning(epochs_target, start_decrease, end_decrease, aggressivity_transition, late_aggressivity):
    """
    Calculates the cumulative pruning factor up to the target epoch.

    Parameters:
        epochs_target (int): Total number of epochs.
        start_decrease (float): Initial pruning decrease factor.
        end_decrease (float): Final pruning decrease factor.
        aggressivity_transition (float): Controls the steepness of the sigmoid transition.
        late_aggressivity (float): Epoch at which the pruning aggressiveness starts to increase.

    Returns:
        float: Cumulative pruning factor at the target epoch.
    """
    log_e_values = [
        np.log(expected_pruning_decrease_at_epoch(
            epoch, start_decrease, end_decrease, aggressivity_transition, late_aggressivity
        ))
        for epoch in range(1, epochs_target + 1)
    ]
    log_product = np.sum(log_e_values)
    # 100 is the initial parameter count (or 100%)
    log_fa = np.log(100) + log_product
    return np.exp(log_fa)

class TrajectoryCalculator:
    def __init__(self, pruning_target, epochs_target, late_aggresivity, aggresivity_transition):

        """
        Initializes the TrajectoryCalculator with the desired pruning parameters.
        Parameters:
            pruning_target (float): The desired pruning factor at the final epoch.
            epochs_target (int): Total number of epochs.
            late_aggressivity (float): Epoch at which the pruning aggressiveness starts to increase.
            aggressivity_transition (float): Controls the steepness of the sigmoid transition.
        """
        self.pruning_target = pruning_target
        self.epochs_target = epochs_target

        self.late_aggressivity = late_aggresivity
        self.aggressivity_transition = aggresivity_transition

        self.END_PRUNING_DECREASE = 0.999
        self.start_pruning_decrease = 0.0

        self.find_start_pruning_decrease()

    def find_start_pruning_decrease(self):
        """
        Finds the optimal starting pruning decrease factor using binary search to meet the pruning target.
        """
        lower_start = 0.0
        upper_start = 0.999

        iteration = 0
        max_iterations = 100

        best_start_decrease = None
        best_end_pruning = None
        MARGIN_ERROR = 1e-6

        while iteration < max_iterations:
            mid_start = (lower_start + upper_start) / 2
            current_end_pruning = expected_pruning(
                self.epochs_target,
                mid_start,
                self.END_PRUNING_DECREASE,
                self.aggressivity_transition,
                self.late_aggressivity
            )

            if abs(current_end_pruning - self.pruning_target) < MARGIN_ERROR:
                best_start_decrease = mid_start
                best_end_pruning = current_end_pruning
                break

            if current_end_pruning < self.pruning_target:
                lower_start = mid_start
            else:
                upper_start = mid_start

            if best_end_pruning is None or abs(current_end_pruning - self.pruning_target) < abs(best_end_pruning - self.pruning_target):
                best_start_decrease = mid_start
                best_end_pruning = current_end_pruning

            iteration += 1

        self.start_pruning_decrease = best_start_decrease
        print(f"\nOptimal start_pruning_decrease: {self.start_pruning_decrease:.6f}")
        print(f"Final pruning at epoch {self.epochs_target}: {best_end_pruning:.6f}")

    def get_start_pruning_decrease(self):
        """
        Returns the optimal starting pruning decrease factor.

        Returns:
            float: Optimal start pruning decrease.
        """
        return self.start_pruning_decrease

    def get_expected_pruning_at_epoch(self, epoch:int):
        return expected_pruning(
            epochs_target=epoch,
            start_decrease=self.start_pruning_decrease,
            end_decrease=self.END_PRUNING_DECREASE,
            aggressivity_transition=self.aggressivity_transition,
            late_aggressivity=self.late_aggressivity
        )



class PruningSchedulerOld:
    def __init__(self, pressure_exponent_constant: float, sparsity_target: float, epochs_target: int, step_size:float):
        self.baseline = 0
        self.pressure_exponent_constant = pressure_exponent_constant
        self.pruning_target = 100 - sparsity_target
        print("SELFT PRUNING target and sparsity ", self.pruning_target, sparsity_target)
        self.epochs_target = epochs_target
        self.streak = 0
        self.step_size = step_size

        self.recorded_states = []

    def _get_previous_decrease(self) -> float:
        """
        Get the percentage decrease from the previous state to the current state
        """
        if len(self.recorded_states) < 2:
            return -1

        if len(self.recorded_states) == 2:
            return self.recorded_states[-1] / self.recorded_states[-2]

        return ((self.recorded_states[-1] / self.recorded_states[-2]) + (self.recorded_states[-2] / self.recorded_states[-3]))/ 2

    def _get_expected_percentage_decrease(self) -> float:
        """
        current * (percentage ** remaining_epochs) = desired
        percentage = (desired / current) ** (1 / remaining_epochs)
        """
        remaining_epochs = self.epochs_target - len(self.recorded_states)
        current_parameters = self.recorded_states[-1]
        return (self.pruning_target / current_parameters) ** (1 / remaining_epochs)


    def get_remaining_epochs(self) -> int:
        return self.epochs_target - len(self.recorded_states)

    def step(self, current_epoch:int, current_sparsity: float) -> None:
        """
        Attempts to predict the final number of weights that will remain after the pruning process, given current pace

        Formula for decreases
        params * decrease = pruned params
        So 0.8 is more aggressive than 0.9
        """
        self.recorded_states.append(current_sparsity)

        if self.get_remaining_epochs() <= 0:
            self.baseline = 0
            return None

        current_decrease = self._get_previous_decrease()
        if current_decrease == -1:
            print("Not enough data to predict")
            return None

        expected_decrease = self._get_expected_percentage_decrease()
        remaining_epochs = self.epochs_target - len(self.recorded_states)

        print(f"Current decrease: {current_decrease * 100:.2f}%, Expected decrease: {expected_decrease * 100:.2f}%")
        if current_decrease > expected_decrease:
            print("Baseline increased !!")
            # expected deviation
            self.baseline += self.step_size + self.streak
            self.streak += self.step_size * 0.1
        else:
            print("Baseline decreased !!")
            self.baseline -= self.step_size/2
            self.streak = 0

    def get_multiplier(self) -> int:
        return self.baseline ** self.pressure_exponent_constant

class PruningSchedulerPolicy2Curve2:
    def __init__(self, pressure_exponent_constant: float, sparsity_target: float, epochs_target: int, step_size:float):
        self.baseline = 0
        self.pressure_exponent_constant = pressure_exponent_constant
        self.pruning_target = 100 - sparsity_target
        self.epochs_target = epochs_target
        self.streak = 0
        self.step_size = step_size

        self.recorded_states = []

    def _get_previous_decrease(self) -> float:
        """
        Get the percentage decrease from the previous state to the current state
        """
        if len(self.recorded_states) < 2:
            return -1

        if len(self.recorded_states) == 2:
            return self.recorded_states[-1] / self.recorded_states[-2]

        return ((self.recorded_states[-1] / self.recorded_states[-2]) + (self.recorded_states[-2] / self.recorded_states[-3]))/ 2

    def _get_expected_percentage_decrease(self) -> float:
        remaining_epochs = self.epochs_target - len(self.recorded_states)
        if remaining_epochs < 1:
            return 1

        current_parameters = self.recorded_states[-1]
        target_parameters = self.pruning_target

        late_aggressivity = remaining_epochs / 2
        aggresivity = 6
        aggressivity_transition = aggresivity / remaining_epochs

        target_relative_to_current = target_parameters / current_parameters*100
        new_trajectory = TrajectoryCalculator(
            pruning_target=target_relative_to_current,
            epochs_target=remaining_epochs,
            late_aggresivity=late_aggressivity,
            aggresivity_transition=aggressivity_transition
        )

        expected_percentage_decrease = new_trajectory.get_expected_pruning_at_epoch(1) / 100
        return expected_percentage_decrease


    def get_remaining_epochs(self) -> int:
        return self.epochs_target - len(self.recorded_states)

    def step(self, current_epoch:int, current_sparsity: float) -> None:
        """
        Attempts to predict the final number of weights that will remain after the pruning process, given current pace

        Formula for decreases
        params * decrease = pruned params
        So 0.8 is more aggressive than 0.9
        """
        self.recorded_states.append(current_sparsity)

        if self.get_remaining_epochs() <= 0:
            self.baseline = 0
            return None

        current_decrease = self._get_previous_decrease()
        if current_decrease == -1:
            print("Not enough data to predict")
            return None

        expected_decrease = self._get_expected_percentage_decrease()
        remaining_epochs = self.epochs_target - len(self.recorded_states)

        print(f"Current decrease: {current_decrease * 100:.2f}%, Expected decrease: {expected_decrease * 100:.2f}%")
        if current_decrease > expected_decrease:
            print("Baseline increased !!")
            # expected deviation
            self.baseline += self.step_size + self.streak
            self.streak += self.step_size * 0.1
        else:
            print("Baseline decreased !!")
            self.baseline -= self.step_size/2
            self.streak = 0
            if self.baseline < 0:
                self.baseline =0

    def get_multiplier(self) -> int:
        return self.baseline ** self.pressure_exponent_constant
