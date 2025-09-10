from dagster import ConfigurableResource


class FraudDatabaseResource(ConfigurableResource):
    """Resource for fraud detection database connections."""

    connection_string: str = "sqlite:///fraud_detection.db"

    def get_connection(self):
        """Get database connection for fraud detection."""
        print(f"Connecting to fraud database: {self.connection_string}")
        return self.connection_string


class FraudModelResource(ConfigurableResource):
    """Resource for fraud detection model configuration."""

    model_type: str = "random_forest"
    max_depth: int = 10
    n_estimators: int = 100

    def get_model_config(self):
        """Get model configuration for fraud detection."""
        return {
            "model_type": self.model_type,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators
        }


class FraudDataResource(ConfigurableResource):
    """Resource for fraud detection data configuration."""

    dataset_url: str = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    test_size: float = 0.2
    random_state: int = 42

    def get_data_config(self):
        """Get data configuration for fraud detection."""
        return {
            "dataset_url": self.dataset_url,
            "test_size": self.test_size,
            "random_state": self.random_state
        }
