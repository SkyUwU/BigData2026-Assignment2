"""Fit train-only feature encoders for sampled temporal subgraphs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import random
from typing import Iterable

import pandas as pd

from feature_policy import FEATURE_COLUMNS_BY_TYPE
from activity_features import USER_ACTIVITY_FEATURE_COLUMNS
from model_input import TIME_DELTA_DAYS_COLUMN, SubgraphModelInput
from temporal_subgraph import EdgeType


@dataclass(frozen=True)
class EncodedNodeFeatures:
    """Numeric and categorical matrices, aligned to the original node rows."""

    numeric: pd.DataFrame
    numeric_missing: pd.DataFrame
    categorical: pd.DataFrame

    @property
    def numeric_with_missing(self) -> pd.DataFrame:
        """Return numeric values and their missing flags as one model input."""
        return pd.concat((self.numeric, self.numeric_missing), axis=1)


@dataclass(frozen=True)
class EncodedSubgraphInput:
    """Model-ready tabular features plus the unchanged sampled graph structure."""

    user_id: int
    cutoff: pd.Timestamp
    node_features: dict[str, EncodedNodeFeatures]
    edge_index: dict[EdgeType, tuple[tuple[int, ...], tuple[int, ...]]]
    seed_node_index: int


@dataclass
class TrainFeatureEncoder:
    """Encode features using statistics and category vocabularies from training only.

    Missing numeric values become the training mean (zero after standardizing),
    with a separate 0/1 missingness column. Categorical ID 0 means unseen at
    training time; ID 1 means missing; IDs from 2 onward are train categories.
    """

    numeric_columns: dict[str, tuple[str, ...]] = field(default_factory=dict)
    categorical_columns: dict[str, tuple[str, ...]] = field(default_factory=dict)
    means: dict[tuple[str, str], float] = field(default_factory=dict)
    scales: dict[tuple[str, str], float] = field(default_factory=dict)
    category_ids: dict[tuple[str, str], dict[str, int]] = field(default_factory=dict)
    fitted: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation for model checkpoints."""
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted feature encoder")
        return {
            "numeric_columns": {key: list(value) for key, value in self.numeric_columns.items()},
            "categorical_columns": {
                key: list(value) for key, value in self.categorical_columns.items()
            },
            "means": [[node, column, value] for (node, column), value in self.means.items()],
            "scales": [[node, column, value] for (node, column), value in self.scales.items()],
            "category_ids": [
                [node, column, categories]
                for (node, column), categories in self.category_ids.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TrainFeatureEncoder":
        """Restore an encoder serialized by :meth:`to_dict`."""
        encoder = cls(
            numeric_columns={key: tuple(value) for key, value in data["numeric_columns"].items()},
            categorical_columns={
                key: tuple(value) for key, value in data["categorical_columns"].items()
            },
            means={(node, column): float(value) for node, column, value in data["means"]},
            scales={(node, column): float(value) for node, column, value in data["scales"]},
            category_ids={
                (node, column): {str(name): int(index) for name, index in categories.items()}
                for node, column, categories in data["category_ids"]
            },
            fitted=True,
        )
        return encoder

    @classmethod
    def from_training(
        cls,
        training_inputs: Iterable[SubgraphModelInput],
        *,
        include_user_activity_features: bool = False,
        activity_columns: tuple[str, ...] = USER_ACTIVITY_FEATURE_COLUMNS,
    ) -> "TrainFeatureEncoder":
        encoder = cls()
        encoder._fit(
            training_inputs,
            include_user_activity_features=include_user_activity_features,
            activity_columns=activity_columns,
        )
        return encoder

    def _fit(
        self,
        training_inputs: Iterable[SubgraphModelInput],
        *,
        include_user_activity_features: bool,
        activity_columns: tuple[str, ...],
    ) -> None:
        numeric_columns: dict[str, tuple[str, ...]] = {}
        categorical_columns: dict[str, tuple[str, ...]] = {}
        sums: dict[tuple[str, str], float] = {}
        sums_sq: dict[tuple[str, str], float] = {}
        counts: dict[tuple[str, str], int] = {}
        categories: dict[tuple[str, str], set[str]] = {}

        for node_type, groups in FEATURE_COLUMNS_BY_TYPE.items():
            numeric_columns[node_type] = (
                *groups["boolean"],
                *groups["numeric"],
                *(
                    activity_columns
                    if node_type == "Users" and include_user_activity_features
                    else ()
                ),
                TIME_DELTA_DAYS_COLUMN,
            )
            categorical_columns[node_type] = groups["categorical"]
            for column in categorical_columns[node_type]:
                categories[(node_type, column)] = set()

        seen_input = False
        for model_input in training_inputs:
            seen_input = True
            for node_type, frame in model_input.node_features.items():
                if node_type not in numeric_columns:
                    raise ValueError(f"Unknown node type in training input: {node_type}")
                expected = set(numeric_columns[node_type]) | set(
                    categorical_columns[node_type]
                )
                missing = expected.difference(frame.columns)
                if missing:
                    raise ValueError(
                        f"{node_type} training features are missing: "
                        + ", ".join(sorted(missing))
                    )
                for column in numeric_columns[node_type]:
                    values = pd.to_numeric(frame[column], errors="coerce").dropna()
                    key = (node_type, column)
                    if not values.empty:
                        sums[key] = sums.get(key, 0.0) + float(values.sum())
                        sums_sq[key] = sums_sq.get(key, 0.0) + float(
                            values.pow(2).sum()
                        )
                        counts[key] = counts.get(key, 0) + int(values.size)
                for column in categorical_columns[node_type]:
                    for value in frame[column].dropna().tolist():
                        categories[(node_type, column)].add(str(value))

        if not seen_input:
            raise ValueError("Cannot fit feature encoder on an empty training input")

        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        for node_type, columns in numeric_columns.items():
            for column in columns:
                key = (node_type, column)
                count = counts.get(key, 0)
                mean = sums.get(key, 0.0) / count if count else 0.0
                variance = sums_sq.get(key, 0.0) / count - mean * mean if count else 0.0
                scale = variance**0.5 if variance > 1e-12 else 1.0
                self.means[key] = mean
                self.scales[key] = scale
        for key, values in categories.items():
            self.category_ids[key] = {
                value: index + 2 for index, value in enumerate(sorted(values))
            }
        self.fitted = True

    def transform(self, model_input: SubgraphModelInput) -> EncodedSubgraphInput:
        if not self.fitted:
            raise RuntimeError("Feature encoder must be fitted on training inputs first")
        encoded: dict[str, EncodedNodeFeatures] = {}
        for node_type, frame in model_input.node_features.items():
            if node_type not in self.numeric_columns:
                raise ValueError(f"Unknown node type in model input: {node_type}")
            numeric_data: dict[str, pd.Series] = {}
            missing_data: dict[str, pd.Series] = {}
            for column in self.numeric_columns[node_type]:
                if column not in frame:
                    raise ValueError(f"{node_type} is missing numeric feature {column!r}")
                # Nullable boolean/integer extension dtypes cannot always fill
                # missing values with a floating-point train mean. Normalize
                # to pandas' nullable float dtype before the imputation step.
                values = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
                missing = values.isna()
                numeric_data[column] = (
                    (values.fillna(self.means[(node_type, column)])
                    - self.means[(node_type, column)])
                    / self.scales[(node_type, column)]
                ).astype("float32")
                missing_data[f"{column}__missing"] = missing.astype("float32")

            categorical_data: dict[str, pd.Series] = {}
            for column in self.categorical_columns[node_type]:
                if column not in frame:
                    raise ValueError(
                        f"{node_type} is missing categorical feature {column!r}"
                    )
                lookup = self.category_ids[(node_type, column)]
                categorical_data[column] = frame[column].map(
                    lambda value: 1 if pd.isna(value) else lookup.get(str(value), 0)
                ).astype("int64")

            encoded[node_type] = EncodedNodeFeatures(
                numeric=pd.DataFrame(numeric_data, index=frame.index),
                numeric_missing=pd.DataFrame(missing_data, index=frame.index),
                categorical=pd.DataFrame(categorical_data, index=frame.index),
            )

        if "Users" not in encoded or encoded["Users"].numeric.empty:
            raise ValueError("Encoded input must contain the target user")
        if not 0 <= model_input.seed_node_index < len(encoded["Users"].numeric):
            raise ValueError("Seed node index is outside the Users feature rows")
        return EncodedSubgraphInput(
            user_id=model_input.user_id,
            cutoff=model_input.cutoff,
            node_features=encoded,
            edge_index=model_input.edge_index,
            seed_node_index=model_input.seed_node_index,
        )

    def apply_training_oov_dropout(
        self,
        encoded_input: EncodedSubgraphInput,
        *,
        probability: float,
        random_seed: int,
    ) -> EncodedSubgraphInput:
        """Replace some known category IDs with OOV ID 0 for training augmentation.

        Call this only for training examples. Missing ID 1 and already-unknown
        ID 0 are preserved. A caller can vary ``random_seed`` by epoch so the
        model sees different OOV examples over training.
        """
        if not self.fitted:
            raise RuntimeError("Feature encoder must be fitted before OOV dropout")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("OOV dropout probability must be between 0 and 1")

        rng = random.Random(random_seed)
        encoded_nodes: dict[str, EncodedNodeFeatures] = {}
        for node_type, node_features in encoded_input.node_features.items():
            categories = node_features.categorical.copy()
            for column in self.categorical_columns[node_type]:
                if column not in categories:
                    raise ValueError(
                        f"{node_type} is missing categorical feature {column!r}"
                    )
                for row_index, category_id in categories[column].items():
                    if category_id >= 2 and rng.random() < probability:
                        categories.at[row_index, column] = 0
            encoded_nodes[node_type] = EncodedNodeFeatures(
                numeric=node_features.numeric,
                numeric_missing=node_features.numeric_missing,
                categorical=categories,
            )

        return replace(encoded_input, node_features=encoded_nodes)
