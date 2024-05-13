"""
File: hopfield.py
Description: Implementation of a Hopfield network for pattern recognition.
Authors: Gizachew Bayness Kassa and Mritunjay Singh
"""
import numpy as np
from typing import Tuple


class HopfieldNetwork:
    """ Implementation of a Hopfield network for pattern recognition.
    """
    def __init__(self, mode='bipolar'):
        """ Initialize the Hopfield network.
        Parameters:
        mode (str): The mode of the network. Must be either 'bipolar' or 'unipolar'.
        """
        self.weights = None  # Initialize weights to None
        self.mode = mode  # 'bipolar' or 'unipolar'
        # self.stored_patterns = None  # Initialize patterns for storing the training patterns
        # self.test_vector = [] # Initialize the test vector

    def train(self, patterns: np.ndarray) -> None:
        """ Train the network with the given patterns.
        Parameters:
        patterns (np.ndarray): The patterns to train the network with.
        """
        self.stored_patterns = patterns # This will help us to decide if the network is stable or not
        if self.mode == 'unipolar':
            patterns = 2 * patterns - 1  # Convert to bipolar
        num_patterns, num_features = patterns.shape
        self.weights = np.zeros((num_features, num_features))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= num_patterns

    def predict(self, pattern: np.ndarray, max_iterations=10) -> Tuple[np.ndarray, int, str]:
        """ Predict the output pattern for the given input pattern.
        Parameters:
        pattern (np.ndarray): The input pattern to predict the output for.
        max_iterations (int): The maximum number of iterations to run the network for.
        Returns:
        Tuple[np.ndarray, int, str]: The output pattern, the number of iterations run, and the information message.
        """
        self.test_vector = pattern
        if self.mode == 'unipolar':
            pattern = 2 * pattern - 1  # Convert to bipolar for processing
        if self.weights is None:
            raise ValueError(
                "The network has not been trained yet. Weights matrix is uninitialized.")
        result = pattern
        iteration = 0
        seen_states = {}
        states_list = []

        while iteration < max_iterations:
            states_list.append(result.copy())  # Store each state in list
            result_key = tuple(result)
            if result_key in seen_states:
                # Cycle detected, return the current state and the state that starts the cycle
                return (f'{self.format_output(states_list[iteration - 1])} and {self.format_output(states_list[iteration])}', iteration, 'Cycle detected between two states')
            seen_states[result_key] = iteration
            result = np.dot(result, self.weights)
            result = np.where(result > 0, 1, -1)  # Bipolar activation
            iteration += 1
            if np.array_equal(result, pattern):
                break
        """"exists = False
        for i in range(len(self.stored_patterns)):
            if np.array_equal(self.stored_patterns, self.test_vector):
                exists = True
                break
        print(exists)
        info = "Converged to input pattern and it is stable." if exists \
            else "Converged to input pattern but it is not stable."
        print(f'test: {self.test_vector}')
        print(f'stored: {self.stored_patterns}') """
        info = "Converged to input pattern."
        if iteration == max_iterations:
            info = "Did not converge to input pattern within the maximum number of iterations."
        return self.format_output(result), iteration, info

    def format_output(self, output) -> np.ndarray:
        """ Format the output pattern based on the mode of the network.
        Parameters:
        output (np.ndarray): The output pattern to format.
        Returns:
        np.ndarray: The formatted output pattern.
        """
        if self.mode == 'unipolar':
            return (output + 1) // 2  # Convert back to unipolar
        return output


# Main script execution
if __name__ == '__main__':
    hn = HopfieldNetwork()
    # Train he model with static patterns
    patterns = np.array([
        [1, -1, 1, -1, 1],
        [1, 1, 1, -1, -1],
        [-1, -1, 1, 1, 1]
    ])
    hn.train(patterns)
    print('The network has trained with static patterns.')
    # Test the model with a pattern
    output, iterations, info = hn.predict(np.array([1, -1, 1, -1, 1]))
    print(f"Output: {output}, Iterations: {iterations}, Info: {info}")
    # Set the mode optional parameter
    hn.mode = 'bipolar' # must be either 'bipolar' or 'unipolar'
    """
    Output:
    The network has trained with static patterns.
    Output: [ 1 -1  1 -1  1], Iterations: 1, Info: Converged to input pattern and it is stable.
    """
