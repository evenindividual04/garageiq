"""Unit tests for ontology.py — Updated for expanded ontology"""
import pytest
from automotive_intent.core.ontology import (
    SERVICE_ONTOLOGY,
    VALID_SYSTEMS,
    get_valid_components,
    get_valid_failure_modes,
    validate_ontology_path,
    get_ontology_formatted,
)


class TestOntologyStructure:
    """Verify the ontology dict is correctly defined."""

    def test_all_systems_present(self):
        expected = {"POWERTRAIN", "ELECTRICAL", "HVAC", "BRAKES", "SUSPENSION", "STEERING", "EXHAUST"}
        assert VALID_SYSTEMS == expected

    def test_powertrain_components(self):
        components = get_valid_components("POWERTRAIN")
        assert components == {"ENGINE", "TRANSMISSION"}

    def test_electrical_components(self):
        components = get_valid_components("ELECTRICAL")
        assert components == {"BATTERY", "STARTER_MOTOR", "ALTERNATOR"}

    def test_hvac_components(self):
        components = get_valid_components("HVAC")
        assert components == {"COMPRESSOR", "BLOWER_MOTOR", "HEATER_CORE"}

    def test_brakes_components(self):
        components = get_valid_components("BRAKES")
        assert components == {"PADS_ROTORS", "BRAKE_FLUID", "ABS"}

    def test_suspension_components(self):
        components = get_valid_components("SUSPENSION")
        assert components == {"SHOCKS_STRUTS", "BALL_JOINTS", "CONTROL_ARMS"}

    def test_steering_components(self):
        components = get_valid_components("STEERING")
        assert components == {"POWER_STEERING", "STEERING_RACK", "TIE_RODS"}

    def test_exhaust_components(self):
        components = get_valid_components("EXHAUST")
        assert components == {"CATALYTIC_CONVERTER", "MUFFLER", "EXHAUST_MANIFOLD"}


class TestOntologyValidation:
    """Test the validate_ontology_path function."""

    def test_valid_path_engine_no_start(self):
        assert validate_ontology_path("POWERTRAIN", "ENGINE", "NO_START") is True

    def test_valid_path_starter_solenoid(self):
        assert validate_ontology_path("ELECTRICAL", "STARTER_MOTOR", "SOLENOID_CLICK") is True

    def test_valid_path_brakes_squealing(self):
        assert validate_ontology_path("BRAKES", "PADS_ROTORS", "SQUEALING") is True

    def test_valid_path_suspension_bouncy(self):
        assert validate_ontology_path("SUSPENSION", "SHOCKS_STRUTS", "BOUNCY_RIDE") is True

    def test_valid_path_steering_stiff(self):
        assert validate_ontology_path("STEERING", "POWER_STEERING", "STIFF_STEERING") is True

    def test_valid_path_exhaust_loud(self):
        assert validate_ontology_path("EXHAUST", "MUFFLER", "LOUD_EXHAUST") is True

    def test_invalid_system(self):
        assert validate_ontology_path("BODY", "DOORS", "STUCK") is False

    def test_invalid_component(self):
        assert validate_ontology_path("POWERTRAIN", "TURBO", "BOOST_LEAK") is False

    def test_invalid_failure_mode(self):
        assert validate_ontology_path("ELECTRICAL", "BATTERY", "EXPLODED") is False

    def test_case_sensitivity(self):
        # Our ontology is uppercase; lowercase should fail
        assert validate_ontology_path("powertrain", "engine", "no_start") is False


class TestHelperFunctions:
    """Test edge cases for helper functions."""

    def test_get_components_invalid_system(self):
        assert get_valid_components("FAKE_SYSTEM") == set()

    def test_get_failure_modes_invalid_system(self):
        assert get_valid_failure_modes("FAKE", "FAKE") == set()

    def test_get_failure_modes_invalid_component(self):
        assert get_valid_failure_modes("POWERTRAIN", "FAKE") == set()

    def test_engine_failure_modes(self):
        modes = get_valid_failure_modes("POWERTRAIN", "ENGINE")
        assert "NO_START" in modes
        assert "ROUGH_IDLE" in modes
        assert "OVERHEATING" in modes
        assert "MISFIRING" in modes  # New mode

    def test_get_ontology_formatted(self):
        formatted = get_ontology_formatted()
        assert "SYSTEM: POWERTRAIN" in formatted
        assert "COMPONENT: ENGINE" in formatted
        assert "FAILURE_MODES: " in formatted
