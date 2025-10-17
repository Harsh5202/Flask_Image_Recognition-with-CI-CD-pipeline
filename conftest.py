"""
Pytest configuration file.
Defines shared fixtures for all test modules.
"""

import pytest
from app import app  # This imports the Flask app for testing


@pytest.fixture
def client():
    """
    Create a Flask test client for testing HTTP endpoints.

    Yields:
        FlaskClient: A test client for the Flask application
    """
    with app.test_client() as test_client:
        yield test_client
