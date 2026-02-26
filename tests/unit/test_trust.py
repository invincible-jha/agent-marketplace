"""Unit tests for agent-marketplace trust modules.

Covers ReputationTracker (reputation.py), Review + ReviewStore (reviews.py),
and ProviderTrustData + TrustScorer (scorer.py).
"""
from __future__ import annotations

import pytest

from agent_marketplace.trust.reputation import ReputationTracker
from agent_marketplace.trust.reviews import Review, ReviewStore
from agent_marketplace.trust.scorer import ProviderTrustData, TrustScorer


# ---------------------------------------------------------------------------
# ReputationTracker — construction
# ---------------------------------------------------------------------------


class TestReputationTrackerInit:
    def test_default_window_size(self) -> None:
        tracker = ReputationTracker()
        assert tracker.window_size == 100

    def test_custom_window_size(self) -> None:
        tracker = ReputationTracker(window_size=10)
        assert tracker.window_size == 10

    def test_zero_window_size_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            ReputationTracker(window_size=0)

    def test_negative_window_size_raises(self) -> None:
        with pytest.raises(ValueError):
            ReputationTracker(window_size=-5)


# ---------------------------------------------------------------------------
# ReputationTracker — record_outcome / get_reputation
# ---------------------------------------------------------------------------


class TestReputationTrackerRecording:
    def setup_method(self) -> None:
        self.tracker = ReputationTracker(window_size=10)

    def test_unknown_provider_returns_zero(self) -> None:
        assert self.tracker.get_reputation("unknown") == 0.0

    def test_all_successes_returns_one(self) -> None:
        for _ in range(5):
            self.tracker.record_outcome("prov-1", success=True)
        assert self.tracker.get_reputation("prov-1") == pytest.approx(1.0)

    def test_all_failures_returns_zero(self) -> None:
        for _ in range(3):
            self.tracker.record_outcome("prov-1", success=False)
        assert self.tracker.get_reputation("prov-1") == pytest.approx(0.0)

    def test_mixed_outcomes_correct_rate(self) -> None:
        self.tracker.record_outcome("prov-1", success=True)
        self.tracker.record_outcome("prov-1", success=False)
        assert self.tracker.get_reputation("prov-1") == pytest.approx(0.5)

    def test_window_evicts_old_records(self) -> None:
        tracker = ReputationTracker(window_size=3)
        # Record 3 successes, then 3 failures — window should only see failures
        for _ in range(3):
            tracker.record_outcome("p", success=True)
        for _ in range(3):
            tracker.record_outcome("p", success=False)
        assert tracker.get_reputation("p") == pytest.approx(0.0)

    def test_multiple_providers_tracked_independently(self) -> None:
        self.tracker.record_outcome("prov-a", success=True)
        self.tracker.record_outcome("prov-b", success=False)
        assert self.tracker.get_reputation("prov-a") == pytest.approx(1.0)
        assert self.tracker.get_reputation("prov-b") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ReputationTracker — inspection
# ---------------------------------------------------------------------------


class TestReputationTrackerInspection:
    def setup_method(self) -> None:
        self.tracker = ReputationTracker()

    def test_total_recorded_unknown_provider(self) -> None:
        assert self.tracker.total_recorded("nobody") == 0

    def test_total_recorded_after_outcomes(self) -> None:
        for _ in range(7):
            self.tracker.record_outcome("prov-1", success=True)
        assert self.tracker.total_recorded("prov-1") == 7

    def test_known_providers_empty(self) -> None:
        assert self.tracker.known_providers() == []

    def test_known_providers_sorted(self) -> None:
        self.tracker.record_outcome("zzz", success=True)
        self.tracker.record_outcome("aaa", success=True)
        providers = self.tracker.known_providers()
        assert providers == sorted(providers)

    def test_reset_clears_outcomes(self) -> None:
        self.tracker.record_outcome("prov-1", success=True)
        self.tracker.reset("prov-1")
        assert self.tracker.total_recorded("prov-1") == 0
        assert self.tracker.get_reputation("prov-1") == 0.0

    def test_reset_unknown_provider_no_error(self) -> None:
        self.tracker.reset("never-recorded")  # Should not raise


# ---------------------------------------------------------------------------
# Review — construction
# ---------------------------------------------------------------------------


class TestReviewConstruction:
    def test_valid_review_created(self) -> None:
        review = Review(reviewer_id="user-1", provider_id="acme", rating=4)
        assert review.reviewer_id == "user-1"
        assert review.provider_id == "acme"
        assert review.rating == 4

    def test_auto_generated_review_id(self) -> None:
        review = Review(reviewer_id="user-1", provider_id="acme", rating=3)
        assert review.review_id
        assert len(review.review_id) > 0

    def test_empty_reviewer_id_raises(self) -> None:
        with pytest.raises(ValueError, match="reviewer_id"):
            Review(reviewer_id="   ", provider_id="acme", rating=3)

    def test_empty_provider_id_raises(self) -> None:
        with pytest.raises(ValueError, match="provider_id"):
            Review(reviewer_id="user-1", provider_id="", rating=3)

    def test_rating_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="rating"):
            Review(reviewer_id="user-1", provider_id="acme", rating=0)

    def test_rating_above_five_raises(self) -> None:
        with pytest.raises(ValueError, match="rating"):
            Review(reviewer_id="user-1", provider_id="acme", rating=6)

    def test_rating_one_is_valid(self) -> None:
        review = Review(reviewer_id="user-1", provider_id="acme", rating=1)
        assert review.rating == 1

    def test_rating_five_is_valid(self) -> None:
        review = Review(reviewer_id="user-1", provider_id="acme", rating=5)
        assert review.rating == 5

    def test_default_text_is_empty(self) -> None:
        review = Review(reviewer_id="user-1", provider_id="acme", rating=3)
        assert review.text == ""

    def test_custom_text_stored(self) -> None:
        review = Review(
            reviewer_id="user-1",
            provider_id="acme",
            rating=5,
            text="Excellent service!",
        )
        assert review.text == "Excellent service!"


# ---------------------------------------------------------------------------
# ReviewStore — CRUD
# ---------------------------------------------------------------------------


class TestReviewStoreCRUD:
    def setup_method(self) -> None:
        self.store = ReviewStore()
        self.review = Review(reviewer_id="user-1", provider_id="acme", rating=4)

    def test_add_and_get(self) -> None:
        self.store.add(self.review)
        retrieved = self.store.get(self.review.review_id)
        assert retrieved.review_id == self.review.review_id

    def test_add_duplicate_raises(self) -> None:
        self.store.add(self.review)
        with pytest.raises(ValueError, match="already exists"):
            self.store.add(self.review)

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.store.get("no-such-review-id")

    def test_update_existing(self) -> None:
        self.store.add(self.review)
        from dataclasses import replace
        updated = replace(self.review, rating=5)
        self.store.update(updated)
        assert self.store.get(self.review.review_id).rating == 5

    def test_update_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.store.update(self.review)

    def test_delete_existing(self) -> None:
        self.store.add(self.review)
        self.store.delete(self.review.review_id)
        with pytest.raises(KeyError):
            self.store.get(self.review.review_id)

    def test_delete_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.store.delete("no-such-id")

    def test_len_empty(self) -> None:
        assert len(self.store) == 0

    def test_len_after_add(self) -> None:
        self.store.add(self.review)
        assert len(self.store) == 1


# ---------------------------------------------------------------------------
# ReviewStore — queries
# ---------------------------------------------------------------------------


class TestReviewStoreQueries:
    def setup_method(self) -> None:
        self.store = ReviewStore()
        self.review_acme_1 = Review(
            reviewer_id="user-1", provider_id="acme", rating=5
        )
        self.review_acme_2 = Review(
            reviewer_id="user-2", provider_id="acme", rating=3
        )
        self.review_beta = Review(
            reviewer_id="user-3", provider_id="beta", rating=4
        )
        self.store.add(self.review_acme_1)
        self.store.add(self.review_acme_2)
        self.store.add(self.review_beta)

    def test_list_for_provider(self) -> None:
        reviews = self.store.list_for_provider("acme")
        assert len(reviews) == 2
        assert all(r.provider_id == "acme" for r in reviews)

    def test_list_for_provider_unknown_returns_empty(self) -> None:
        assert self.store.list_for_provider("nonexistent") == []

    def test_average_rating_computed(self) -> None:
        avg = self.store.average_rating("acme")
        assert avg == pytest.approx(4.0)  # (5 + 3) / 2

    def test_average_rating_no_reviews_returns_zero(self) -> None:
        assert self.store.average_rating("nobody") == pytest.approx(0.0)

    def test_average_rating_single_review(self) -> None:
        assert self.store.average_rating("beta") == pytest.approx(4.0)

    def test_count_for_provider(self) -> None:
        assert self.store.count_for_provider("acme") == 2
        assert self.store.count_for_provider("beta") == 1

    def test_count_for_unknown_provider_is_zero(self) -> None:
        assert self.store.count_for_provider("no-provider") == 0

    def test_list_all_returns_all_reviews(self) -> None:
        all_reviews = self.store.list_all()
        assert len(all_reviews) == 3


# ---------------------------------------------------------------------------
# ProviderTrustData — validation
# ---------------------------------------------------------------------------


class TestProviderTrustDataValidation:
    def test_valid_data_accepted(self) -> None:
        data = ProviderTrustData(
            provider_id="prov-1",
            registration_age_days=180.0,
            usage_count=500,
            success_rate=0.95,
            review_score=4.2,
        )
        assert data.provider_id == "prov-1"

    def test_negative_age_raises(self) -> None:
        with pytest.raises(ValueError, match="registration_age_days"):
            ProviderTrustData(
                provider_id="p",
                registration_age_days=-1.0,
                usage_count=0,
                success_rate=0.5,
                review_score=0.0,
            )

    def test_negative_usage_count_raises(self) -> None:
        with pytest.raises(ValueError, match="usage_count"):
            ProviderTrustData(
                provider_id="p",
                registration_age_days=10.0,
                usage_count=-1,
                success_rate=0.5,
                review_score=0.0,
            )

    def test_success_rate_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="success_rate"):
            ProviderTrustData(
                provider_id="p",
                registration_age_days=10.0,
                usage_count=0,
                success_rate=1.1,
                review_score=0.0,
            )

    def test_success_rate_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="success_rate"):
            ProviderTrustData(
                provider_id="p",
                registration_age_days=10.0,
                usage_count=0,
                success_rate=-0.1,
                review_score=0.0,
            )

    def test_review_score_above_five_raises(self) -> None:
        with pytest.raises(ValueError, match="review_score"):
            ProviderTrustData(
                provider_id="p",
                registration_age_days=10.0,
                usage_count=0,
                success_rate=0.5,
                review_score=5.1,
            )

    def test_review_score_zero_is_valid(self) -> None:
        data = ProviderTrustData(
            provider_id="p",
            registration_age_days=0.0,
            usage_count=0,
            success_rate=0.0,
            review_score=0.0,
        )
        assert data.review_score == 0.0


# ---------------------------------------------------------------------------
# TrustScorer — construction
# ---------------------------------------------------------------------------


class TestTrustScorerInit:
    def test_default_weights_accepted(self) -> None:
        scorer = TrustScorer()
        assert scorer is not None

    def test_bad_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="Weights"):
            TrustScorer(
                age_weight=0.5,
                usage_weight=0.5,
                success_rate_weight=0.5,
                review_weight=0.5,
            )

    def test_zero_age_saturation_raises(self) -> None:
        with pytest.raises(ValueError, match="age_saturation_days"):
            TrustScorer(
                age_weight=0.20,
                usage_weight=0.30,
                success_rate_weight=0.30,
                review_weight=0.20,
                age_saturation_days=0.0,
            )

    def test_zero_usage_saturation_raises(self) -> None:
        with pytest.raises(ValueError, match="usage_saturation"):
            TrustScorer(
                age_weight=0.20,
                usage_weight=0.30,
                success_rate_weight=0.30,
                review_weight=0.20,
                usage_saturation=0,
            )


# ---------------------------------------------------------------------------
# TrustScorer — score()
# ---------------------------------------------------------------------------


def _make_data(
    age_days: float = 180.0,
    usage_count: int = 500,
    success_rate: float = 0.9,
    review_score: float = 4.0,
) -> ProviderTrustData:
    return ProviderTrustData(
        provider_id="prov-test",
        registration_age_days=age_days,
        usage_count=usage_count,
        success_rate=success_rate,
        review_score=review_score,
    )


class TestTrustScorerScore:
    def setup_method(self) -> None:
        self.scorer = TrustScorer()

    def test_score_in_valid_range(self) -> None:
        data = _make_data()
        score = self.scorer.score(data)
        assert 0.0 <= score <= 1.0

    def test_perfect_provider_high_score(self) -> None:
        data = _make_data(
            age_days=365.0,
            usage_count=1000,
            success_rate=1.0,
            review_score=5.0,
        )
        score = self.scorer.score(data)
        assert score > 0.8

    def test_brand_new_provider_low_score(self) -> None:
        data = _make_data(
            age_days=0.0,
            usage_count=0,
            success_rate=0.0,
            review_score=0.0,
        )
        score = self.scorer.score(data)
        assert score < 0.5

    def test_older_provider_higher_score_than_newer(self) -> None:
        old_data = _make_data(age_days=300.0, usage_count=100, success_rate=0.8, review_score=4.0)
        new_data = _make_data(age_days=1.0, usage_count=100, success_rate=0.8, review_score=4.0)
        assert self.scorer.score(old_data) > self.scorer.score(new_data)

    def test_higher_success_rate_higher_score(self) -> None:
        high_rate = _make_data(success_rate=0.99)
        low_rate = _make_data(success_rate=0.10)
        assert self.scorer.score(high_rate) > self.scorer.score(low_rate)

    def test_no_review_treated_as_neutral(self) -> None:
        data_no_review = _make_data(review_score=0.0)
        # review_score=0 -> normalise returns 0.5 (neutral, not 0)
        score = self.scorer.score(data_no_review)
        assert score > 0.0

    def test_score_clamped_to_one(self) -> None:
        data = _make_data(
            age_days=9999.0,
            usage_count=999999,
            success_rate=1.0,
            review_score=5.0,
        )
        score = self.scorer.score(data)
        assert score <= 1.0

    def test_score_clamped_to_zero(self) -> None:
        data = _make_data(age_days=0.0, usage_count=0, success_rate=0.0, review_score=1.0)
        score = self.scorer.score(data)
        assert score >= 0.0

    def test_normalise_age_zero_returns_zero(self) -> None:
        assert self.scorer._normalise_age(0.0) == pytest.approx(0.0)

    def test_normalise_usage_zero_returns_zero(self) -> None:
        assert self.scorer._normalise_usage(0) == pytest.approx(0.0)

    def test_normalise_review_zero_returns_half(self) -> None:
        assert TrustScorer._normalise_review(0.0) == pytest.approx(0.5)

    def test_normalise_review_five_returns_one(self) -> None:
        assert TrustScorer._normalise_review(5.0) == pytest.approx(1.0)

    def test_normalise_review_one_returns_zero(self) -> None:
        assert TrustScorer._normalise_review(1.0) == pytest.approx(0.0)
