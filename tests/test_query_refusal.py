"""Tests for bonus/query_refusal.py — PII, legal, medical screening."""

from __future__ import annotations

from bonus.query_refusal import (
    LEGAL_DISCLAIMER_MESSAGE,
    MEDICAL_DISCLAIMER_MESSAGE,
    PII_REFUSAL_MESSAGE,
    screen_query,
)


# ---------------------------------------------------------------------------
# PII detection
# ---------------------------------------------------------------------------


class TestPIIDetection:
    def test_email(self):
        r = screen_query("Contact me at john@example.com please")
        assert r.refused
        assert "email" in r.reason.lower()
        assert r.message == PII_REFUSAL_MESSAGE

    def test_phone_dashes(self):
        r = screen_query("Call 555-123-4567")
        assert r.refused
        assert "phone" in r.reason.lower()

    def test_phone_dots(self):
        r = screen_query("My number is 555.123.4567")
        assert r.refused

    def test_ssn(self):
        r = screen_query("My SSN is 123-45-6789")
        assert r.refused
        assert "Social Security" in r.reason

    def test_ssn_no_dashes(self):
        r = screen_query("SSN 123456789")
        assert r.refused

    def test_credit_card(self):
        r = screen_query("Card number 4111 1111 1111 1111")
        assert r.refused
        assert "credit card" in r.reason.lower()

    def test_email_embedded_in_text(self):
        r = screen_query("I found this user user@corp.io in the logs")
        assert r.refused


# ---------------------------------------------------------------------------
# Legal advice detection
# ---------------------------------------------------------------------------


class TestLegalDetection:
    def test_legal_advice(self):
        r = screen_query("I need legal advice about my lease")
        assert r.refused
        assert "Legal" in r.reason
        assert r.message == LEGAL_DISCLAIMER_MESSAGE

    def test_can_i_sue(self):
        r = screen_query("Can I sue my landlord?")
        assert r.refused

    def test_lawyer(self):
        r = screen_query("Should I hire a lawyer for this?")
        assert r.refused

    def test_am_i_liable(self):
        r = screen_query("Am I liable for the damages?")
        assert r.refused

    def test_legal_rights(self):
        r = screen_query("What are my legal rights here?")
        assert r.refused


# ---------------------------------------------------------------------------
# Medical advice detection
# ---------------------------------------------------------------------------


class TestMedicalDetection:
    def test_should_i_take(self):
        r = screen_query("Should I take ibuprofen for this?")
        assert r.refused
        assert "Medical" in r.reason
        assert r.message == MEDICAL_DISCLAIMER_MESSAGE

    def test_symptoms_of(self):
        r = screen_query("What are the symptoms of diabetes?")
        assert r.refused

    def test_is_it_safe_to_take(self):
        r = screen_query("Is it safe to take aspirin with alcohol?")
        assert r.refused

    def test_medical_advice(self):
        r = screen_query("I need medical advice")
        assert r.refused

    def test_stop_taking(self):
        r = screen_query("Can I stop taking my medication?")
        assert r.refused

    def test_am_i_sick(self):
        r = screen_query("Am I sick?")
        assert r.refused


# ---------------------------------------------------------------------------
# Clean queries (should pass through)
# ---------------------------------------------------------------------------


class TestCleanQueries:
    def test_general_question(self):
        r = screen_query("What is in-context learning?")
        assert not r.refused

    def test_greeting(self):
        r = screen_query("hello")
        assert not r.refused

    def test_technical_question(self):
        r = screen_query("Explain the architecture of transformers")
        assert not r.refused

    def test_academic_diagnosis(self):
        """Academic use of 'diagnosis' should not trigger medical refusal."""
        r = screen_query("What does the paper say about diagnosis accuracy?")
        assert not r.refused

    def test_academic_prescription(self):
        r = screen_query("How does the model handle prescription drug names as entities?")
        assert not r.refused

    def test_empty_query(self):
        r = screen_query("")
        assert not r.refused
