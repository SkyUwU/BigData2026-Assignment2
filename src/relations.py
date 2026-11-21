"""Convert filtered relational foreign keys into typed graph edges."""

from __future__ import annotations

import pandas as pd


PRIMARY_KEYS = {
    "Users": "user_id",
    "Repositories": "repo_id",
    "PullRequests": "pr_id",
    "Issues": "issue_id",
    "IssueComments": "comment_id",
    "PRReviews": "review_id",
    "PRReviewComments": "comment_id",
    "Pushes": "push_id",
}

# (table containing the FK, FK column, referenced table, relation name)
FOREIGN_KEYS = (
    ("PullRequests", "repo_id", "Repositories", "targets_repo"),
    ("PullRequests", "author_id", "Users", "authored_by"),
    ("Issues", "repo_id", "Repositories", "in_repo"),
    ("Issues", "author_id", "Users", "authored_by"),
    ("Issues", "assignee_id", "Users", "assigned_to"),
    ("Issues", "pull_request_id", "PullRequests", "conversation_for"),
    ("IssueComments", "issue_id", "Issues", "comments_on"),
    ("IssueComments", "repo_id", "Repositories", "in_repo"),
    ("IssueComments", "author_id", "Users", "authored_by"),
    ("PRReviews", "pr_id", "PullRequests", "reviews"),
    ("PRReviews", "repo_id", "Repositories", "in_repo"),
    ("PRReviews", "reviewer_id", "Users", "reviewed_by"),
    ("PRReviewComments", "pr_id", "PullRequests", "comments_on"),
    ("PRReviewComments", "repo_id", "Repositories", "in_repo"),
    ("PRReviewComments", "author_id", "Users", "authored_by"),
    ("PRReviewComments", "pull_request_review_id", "PRReviews", "belongs_to_review"),
    ("PRReviewComments", "in_reply_to_id", "PRReviewComments", "replies_to"),
    ("Pushes", "user_id", "Users", "pushed_by"),
    ("Pushes", "repo_id", "Repositories", "pushes_to"),
)


def build_relation_edges(
    tables: dict[str, pd.DataFrame],
) -> dict[tuple[str, str, str], tuple[list[int], list[int]]]:
    """Return typed, locally indexed edges for FK links present in the snapshot.

    Rows that point to null or to a node excluded by the time cutoff are skipped.
    The returned integer indices refer to row positions in each filtered table,
    not to GitHub IDs or the database's row-number primary keys.
    """
    id_to_local_index: dict[str, dict[int, int]] = {}
    for table_name, primary_key in PRIMARY_KEYS.items():
        table = tables[table_name]
        id_to_local_index[table_name] = {
            int(entity_id): local_index
            for local_index, entity_id in enumerate(table[primary_key])
        }

    edges: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}
    for source_table, fk_column, target_table, relation_name in FOREIGN_KEYS:
        source_ids: list[int] = []
        target_ids: list[int] = []
        target_lookup = id_to_local_index[target_table]
        source_lookup = id_to_local_index[source_table]
        source_pk = PRIMARY_KEYS[source_table]

        for entity_id, foreign_id in zip(tables[source_table][source_pk], tables[source_table][fk_column]):
            if pd.isna(foreign_id):
                continue

            source_index = source_lookup[int(entity_id)]
            target_index = target_lookup.get(int(foreign_id))
            if target_index is None:
                # The referenced row may not have been available at this cutoff.
                continue

            source_ids.append(source_index)
            target_ids.append(target_index)

        edge_type = (source_table, relation_name, target_table)
        edges[edge_type] = (source_ids, target_ids)
        reverse_edge_type = (target_table, f"rev_{relation_name}", source_table)
        edges[reverse_edge_type] = (target_ids, source_ids)

    return edges
