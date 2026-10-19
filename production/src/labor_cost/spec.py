"""Analyst-facing declarations for the supported workforce model family."""

import json
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from hashlib import sha256

from .expressions import Expr


@dataclass(frozen=True)
class InputSpec:
    """Bind a row input to its units, representation and atomic driver.

    ``aligned`` inputs retain employee identity. ``payment`` inputs are
    actual row amounts converted to per-exposure cell means. ``policy``
    identifies the fixed regional coefficient, not an attribution player.
    """

    name: str
    source: str
    label: str
    representation: str
    units: str
    constraint: str = "signed"
    driver: str | None = None

    @property
    def driver_id(self):
        """Return the stable state/player identifier."""
        return self.driver or self.name


@dataclass(frozen=True)
class DistributionSpec:
    """Declare one entire probability object and its conditioning context."""

    driver: str
    dimensions: tuple[str, ...]
    given: tuple[str, ...]
    label: str

    def __post_init__(self):
        """Freeze collections supplied by analyst callers."""
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "given", tuple(self.given))


@dataclass(frozen=True)
class WorkforceStructure:
    """Declare supported region-first composition and nested games.

    Multiple grade columns define one joint category within region,
    switched as a complete conditional matrix. An optional region column
    identifies actual regions under a consistent coefficient schedule.
    The default retains coefficient bands, which are not geography.
    """

    grade_columns: tuple[str, ...] = ("grade",)
    region_column: str | None = None
    distributions: tuple[DistributionSpec, ...] = (
        DistributionSpec("team_mix", ("team",), (), "Team mix"),
        DistributionSpec(
            "contract_type_mix",
            ("contract_type",),
            ("team",),
            "Contract type mix",
        ),
        DistributionSpec(
            "regional_mix",
            ("region",),
            ("team", "permanent"),
            "Regional coefficient-band mix",
        ),
        DistributionSpec(
            "grade_mix_within_region",
            ("grade",),
            ("team", "permanent", "region"),
            "Grade composition within region",
        ),
    )
    hierarchy: str = "organization/team/contract/permanent"
    propagation: str = "affine_child"
    membership_label: str = "Within-cell workforce mix — base pay"
    fallback_order: tuple[str, ...] = ("grade", "region", "global")

    def __post_init__(self):
        """Freeze structural collections to keep model identities stable."""
        for name in ("grade_columns", "distributions", "fallback_order"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class ModelSpec:
    """A versioned payroll formula, state representation and game structure.

    Models use the production employee-month-employment-type source grain.
    Completion and exposure defaults are explicit and can be recorded as
    run overrides. New mathematical primitives require engine extensions.
    """

    model_id: str
    version: str
    inputs: tuple[InputSpec, ...] = ()
    formula: Expr | None = None
    structure: WorkforceStructure = dc_field(
        default_factory=WorkforceStructure
    )
    missing_economics: str = "carry_observed"
    exposure_column: str | None = None
    graph: object | None = None

    def __post_init__(self):
        """Retain an immutable collection of input declarations."""
        object.__setattr__(self, "inputs", tuple(self.inputs))

    def to_dict(self):
        """Serialize declarations without executable Python objects."""
        if self.graph is not None:
            return json.loads(
                json.dumps(
                    {
                        "model_id": self.model_id,
                        "version": self.version,
                        "schema_version": 2,
                        "graph": self.graph.to_dict(),
                    },
                    allow_nan=False,
                )
            )
        value = asdict(self)
        value.pop("graph")
        value["formula"] = self.formula.to_dict()
        return json.loads(json.dumps(value, allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        """Read a model specification previously written by ``to_dict``."""
        value = dict(value)
        if value.get("schema_version") == 2:
            from .graph import ModelGraph

            return cls(
                value["model_id"],
                value["version"],
                graph=ModelGraph.from_dict(value["graph"]),
            )
        if "schema_version" in value:
            raise ValueError("Unsupported model schema version")
        value["inputs"] = tuple(InputSpec(**x) for x in value["inputs"])
        value["formula"] = Expr.from_dict(value["formula"])
        structure = dict(value.get("structure", {}))
        if "distributions" in structure:
            structure["distributions"] = tuple(
                DistributionSpec(
                    x["driver"],
                    tuple(x["dimensions"]),
                    tuple(x["given"]),
                    x["label"],
                )
                for x in structure["distributions"]
            )
        for key in ("grade_columns", "fallback_order"):
            if key in structure:
                structure[key] = tuple(structure[key])
        value["structure"] = WorkforceStructure(**structure)
        return cls(**value)

    @property
    def fingerprint(self):
        """Identify model content independently of its human version label."""
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return sha256(payload.encode()).hexdigest()
