"""
Test module for integration testing - sad path scenarios.
Tests verify error handling when invalid or missing data is provided.
"""

# test_integration_sad.py

import pytest
from io import BytesIO


def test_missing_file(client):
    """Test the prediction route with a missing file."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data  # Check if the error message is displayed
