"""Capability verification rule pipeline.

Design
------
Each rule implements the :class:`VerificationRule` ABC and returns a
(passed: bool, message: str) tuple.  The :class:`CapabilityVerifier`
runs all configured rules against an :class:`AgentCapability` and
accumulates the results into a :class:`VerificationResult`.

Rules are stateless and can be instantiated once and reused.

Usage
-----
::

    from agent_marketplace.verification import CapabilityVerifier, TrustLevelRule

    verifier = CapabilityVerifier([
        TrustLevelRule(min_trust=0.7),
        CompletenessRule(),
        QualityMetricsRule(require_verified=True),
    ])
    result = verifier.verify(capability)
    if not result.passed:
        print(result.failures)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_marketplace.schema.capability import AgentCapability


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    """Immutable outcome of a capability verification run.

    Parameters
    ----------
    capability_id:
        The capability that was verified.
    passed:
        True when all rules passed.
    failures:
        Human-readable failure messages from rules that did not pass.
    warnings:
        Non-blocking advisory messages.
    rules_run:
        Names of rules that were executed.
    verified_at:
        UTC timestamp of the verification.
    metadata:
        Additional annotations.
    """

    capability_id: str
    passed: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    rules_run: tuple[str, ...]
    verified_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def failure_count(self) -> int:
        """Number of rule failures."""
        return len(self.failures)

    @property
    def warning_count(self) -> int:
        """Number of advisory warnings."""
        return len(self.warnings)

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"[{status}] capability={self.capability_id} "
            f"rules={len(self.rules_run)} "
            f"failures={self.failure_count} "
            f"warnings={self.warning_count}"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""
        return {
            "capability_id": self.capability_id,
            "passed": self.passed,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "rules_run": list(self.rules_run),
            "verified_at": self.verified_at.isoformat(),
            "failure_count": self.failure_count,
            "warning_count": self.warning_count,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# VerificationRule — ABC
# ---------------------------------------------------------------------------


class VerificationRule(ABC):
    """Abstract base class for a single verification rule.

    Each rule inspects one aspect of an :class:`AgentCapability` and
    returns a result tuple.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short rule identifier (e.g. ``"completeness"``)."""

    @abstractmethod
    def check(self, capability: AgentCapability) -> tuple[bool, str]:
        """Run the rule against *capability*.

        Parameters
        ----------
        capability:
            The capability to inspect.

        Returns
        -------
        tuple[bool, str]
            ``(passed, message)`` — *message* is empty when *passed* is True,
            otherwise it contains a human-readable failure reason.
        """

    @property
    def is_warning_only(self) -> bool:
        """If True, failures are treated as warnings and do not block listing.

        Defaults to False.
        """
        return False


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------


class CompletenessRule(VerificationRule):
    """Verify that required fields are non-empty.

    Checks: description, input_types (non-empty list), output_type.

    Parameters
    ----------
    require_tags:
        If True, the capability must have at least one tag.  Default False.
    require_latency_profile:
        If True, at least the p50_ms must be > 0.  Default False.
    """

    def __init__(
        self,
        require_tags: bool = False,
        require_latency_profile: bool = False,
    ) -> None:
        self._require_tags = require_tags
        self._require_latency_profile = require_latency_profile

    @property
    def name(self) -> str:
        return "completeness"

    def check(self, capability: AgentCapability) -> tuple[bool, str]:
        issues: list[str] = []

        if not capability.description.strip():
            issues.append("description is empty")

        if not capability.input_types:
            issues.append("input_types is empty")

        if not capability.output_type.strip():
            issues.append("output_type is empty")

        if self._require_tags and not capability.tags:
            issues.append("no tags specified")

        if self._require_latency_profile and capability.latency.p50_ms == 0.0:
            issues.append("latency.p50_ms is 0 (no latency data)")

        if issues:
            return False, "Completeness failures: " + "; ".join(issues)
        return True, ""


class QualityMetricsRule(VerificationRule):
    """Verify that quality metrics meet minimum requirements.

    Parameters
    ----------
    require_verified:
        If True, ``quality_metrics.verified`` must be True.
    min_metric_value:
        Any numeric metric in ``quality_metrics.metrics`` must be >= this.
        Default 0.0 (no minimum).
    require_metrics:
        List of metric names that must be present in ``quality_metrics.metrics``.
    is_warning_only:
        If True, failures from this rule are advisory only.  Default False.
    """

    def __init__(
        self,
        require_verified: bool = False,
        min_metric_value: float = 0.0,
        require_metrics: list[str] | None = None,
        warning_only: bool = False,
    ) -> None:
        self._require_verified = require_verified
        self._min_metric_value = min_metric_value
        self._require_metrics = list(require_metrics or [])
        self._warning_only = warning_only

    @property
    def name(self) -> str:
        return "quality_metrics"

    @property
    def is_warning_only(self) -> bool:
        return self._warning_only

    def check(self, capability: AgentCapability) -> tuple[bool, str]:
        issues: list[str] = []
        qm = capability.quality_metrics

        if self._require_verified and not qm.verified:
            issues.append("quality_metrics.verified is False")

        if self._min_metric_value > 0.0:
            for metric_name, value in qm.metrics.items():
                if value < self._min_metric_value:
                    issues.append(
                        f"metric {metric_name!r}={value} < min {self._min_metric_value}"
                    )

        for required_metric in self._require_metrics:
            if required_metric not in qm.metrics:
                issues.append(f"required metric {required_metric!r} is missing")

        if issues:
            return False, "Quality metrics failures: " + "; ".join(issues)
        return True, ""


class TrustLevelRule(VerificationRule):
    """Verify that the capability's trust_level meets a minimum threshold.

    Parameters
    ----------
    min_trust:
        Minimum required trust level in [0.0, 1.0].  Default 0.5.
    is_warning_only:
        If True, a low trust level is reported as a warning only.
    """

    def __init__(
        self,
        min_trust: float = 0.5,
        warning_only: bool = False,
    ) -> None:
        if not (0.0 <= min_trust <= 1.0):
            raise ValueError(
                f"min_trust must be in [0.0, 1.0], got {min_trust!r}."
            )
        self._min_trust = min_trust
        self._warning_only = warning_only

    @property
    def name(self) -> str:
        return "trust_level"

    @property
    def is_warning_only(self) -> bool:
        return self._warning_only

    def check(self, capability: AgentCapability) -> tuple[bool, str]:
        if capability.trust_level < self._min_trust:
            return (
                False,
                f"trust_level {capability.trust_level:.3f} < required {self._min_trust:.3f}",
            )
        return True, ""


class SupportedFrameworksRule(VerificationRule):
    """Verify that the capability supports at least one required framework.

    Parameters
    ----------
    required_frameworks:
        The capability must support at least one of these.
    is_warning_only:
        Treat failures as warnings.  Default False.
    """

    def __init__(
        self,
        required_frameworks: list[str],
        warning_only: bool = False,
    ) -> None:
        if not required_frameworks:
            raise ValueError("required_frameworks must not be empty.")
        self._required = set(required_frameworks)
        self._warning_only = warning_only

    @property
    def name(self) -> str:
        return "supported_frameworks"

    @property
    def is_warning_only(self) -> bool:
        return self._warning_only

    def check(self, capability: AgentCapability) -> tuple[bool, str]:
        supported = set(capability.supported_frameworks)
        if not (supported & self._required):
            return (
                False,
                f"Capability does not support any of the required frameworks "
                f"{sorted(self._required)!r}. "
                f"Supported: {sorted(supported)!r}.",
            )
        return True, ""


# ---------------------------------------------------------------------------
# CapabilityVerifier
# ---------------------------------------------------------------------------


class CapabilityVerifier:
    """Orchestrate a configurable pipeline of verification rules.

    Parameters
    ----------
    rules:
        Ordered list of :class:`VerificationRule` instances to apply.
        If None, a sensible default set is used.
    stop_on_first_failure:
        If True, stop running rules after the first blocking failure.
        Default False (run all rules and collect all failures).

    Example
    -------
    ::

        verifier = CapabilityVerifier()
        result = verifier.verify(capability)
        if not result.passed:
            print(result.failures)
    """

    _DEFAULT_RULES: list[VerificationRule] = []  # populated after class definition

    def __init__(
        self,
        rules: list[VerificationRule] | None = None,
        stop_on_first_failure: bool = False,
    ) -> None:
        self._rules: list[VerificationRule] = (
            rules if rules is not None else list(self._DEFAULT_RULES)
        )
        self._stop_on_first_failure = stop_on_first_failure

    def add_rule(self, rule: VerificationRule) -> None:
        """Append a rule to the pipeline.

        Parameters
        ----------
        rule:
            The rule to add.
        """
        self._rules.append(rule)

    def verify(self, capability: AgentCapability) -> VerificationResult:
        """Run all rules against *capability* and return the result.

        Parameters
        ----------
        capability:
            The :class:`AgentCapability` to verify.

        Returns
        -------
        VerificationResult
            Aggregated result from all rules.
        """
        failures: list[str] = []
        warnings: list[str] = []
        rules_run: list[str] = []

        for rule in self._rules:
            rules_run.append(rule.name)
            passed, message = rule.check(capability)
            if not passed:
                if rule.is_warning_only:
                    warnings.append(f"[{rule.name}] {message}")
                else:
                    failures.append(f"[{rule.name}] {message}")
                    if self._stop_on_first_failure:
                        break

        overall_passed = len(failures) == 0

        return VerificationResult(
            capability_id=capability.capability_id,
            passed=overall_passed,
            failures=tuple(failures),
            warnings=tuple(warnings),
            rules_run=tuple(rules_run),
            verified_at=datetime.now(timezone.utc),
        )

    def verify_batch(
        self, capabilities: list[AgentCapability]
    ) -> list[VerificationResult]:
        """Run verification on a list of capabilities.

        Parameters
        ----------
        capabilities:
            Capabilities to verify.

        Returns
        -------
        list[VerificationResult]
            One result per capability, in the same order.
        """
        return [self.verify(cap) for cap in capabilities]

    @property
    def rule_count(self) -> int:
        """Number of configured rules."""
        return len(self._rules)

    @property
    def rule_names(self) -> list[str]:
        """Names of all configured rules in order."""
        return [r.name for r in self._rules]


# Populate default rules after class is defined
CapabilityVerifier._DEFAULT_RULES = [
    CompletenessRule(),
    TrustLevelRule(min_trust=0.0),  # advisory — any trust level accepted by default
]
