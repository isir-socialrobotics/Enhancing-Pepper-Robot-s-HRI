
class KalmanFilter1D:
    # A 1D Kalman filter implementation

    def __init__(self, initial_state, initial_uncertainty, process_variance, measurement_variance):
        self.state = initial_state
        self.uncertainty = initial_uncertainty
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def predict(self):
        # Prediction step: in 1D Kalman filter, state doesn't change
        self.state = self.state
        self.uncertainty += self.process_variance

    def update(self, measurement):
        # Update step: adjust state based on new measurement
        kalman_gain = self.uncertainty / (self.uncertainty + self.measurement_variance)
        self.state += kalman_gain * (measurement - self.state)
        self.uncertainty = (1 - kalman_gain) * self.uncertainty

    def get_state(self):
        return self.state