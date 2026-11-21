"""Allowlist of original table attributes safe to consider as model inputs.

Fields are selected from data/db/SCHEMA.md. Cutoff-specific derived activity
summaries are defined separately in activity_features.py. Row availability is
still enforced by load_snapshot() in data.py.
"""

FEATURE_COLUMNS_BY_TYPE = {
    "Users": {
        "boolean": (),
        "numeric": (),
        "categorical": (),
    },
    "Repositories": {
        "boolean": (),
        "numeric": (),
        "categorical": (),
    },
    "PullRequests": {
        # is_cross_repo is derived from repository IDs present when the PR row
        # is first seen. draft is omitted because its point-in-time availability
        # is not explicit enough for the conservative first version.
        "boolean": ("is_cross_repo",),
        "numeric": (),
        "categorical": (),
    },
    "Issues": {
        # These are recorded from the first event about the issue.
        "boolean": ("is_pull_request", "locked"),
        "numeric": ("num_labels",),
        "categorical": ("issue_type", "created_via_app"),
    },
    "IssueComments": {
        # Both values come from the comment event itself.
        "boolean": ("is_pull_request",),
        "numeric": (),
        "categorical": ("via_app",),
    },
    "PRReviews": {
        # state is the review state as submitted.
        "boolean": (),
        "numeric": (),
        "categorical": ("state",),
    },
    "PRReviewComments": {
        # These describe the comment when it was posted; null positions remain
        # missing values and will need an explicit missing-value encoding.
        "boolean": (),
        "numeric": ("position", "original_position"),
        "categorical": ("file_extension", "subject_type"),
    },
    "Pushes": {
        # Both values are derived from the push event payload.
        "boolean": ("is_new_ref",),
        "numeric": (),
        "categorical": ("ref_type",),
    },
}


def feature_columns(table_name: str) -> tuple[str, ...]:
    """Return the allowlisted feature columns for one table."""
    groups = FEATURE_COLUMNS_BY_TYPE[table_name]
    return tuple(
        column
        for feature_type in ("boolean", "numeric", "categorical")
        for column in groups[feature_type]
    )
