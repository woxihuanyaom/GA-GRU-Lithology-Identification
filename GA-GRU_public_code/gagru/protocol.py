from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import ExternalTestLockedError, ProtocolError


PROTOCOL_JSON = "实验协议_冻结配置_v1.0.json"
WELL_MANIFEST = "井级用途_冻结版_v1.0.csv"
FOLD_MANIFEST = "外层与内层划分_冻结版_v1.0.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_wells(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split("|") if item)


def resolve_protocol_dir(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_path = os.environ.get("GAGRU_PROTOCOL_DIR")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path(__file__).resolve().parents[1] / "experiment_protocol_v1",
            Path(__file__).resolve().parents[2] / "experiment_protocol_v1",
        ]
    )

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        checked.append(str(resolved))
        if all((resolved / name).is_file() for name in (PROTOCOL_JSON, WELL_MANIFEST, FOLD_MANIFEST)):
            return resolved
    raise ProtocolError("Frozen protocol directory not found. Checked: " + "; ".join(checked))


@dataclass(frozen=True)
class WellSpec:
    well_id: str
    role: str
    source_relative_path: str
    rows_5curve: int
    continuous_segments: int
    boundary_rows: int
    class_ids_present: tuple[int, ...]
    outer_fold: int | None
    external_locked: bool
    reason: str


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    outer_test_well: str
    inner_validation_wells: tuple[str, ...]
    inner_training_wells: tuple[str, ...]
    outer_training_wells: tuple[str, ...]


@dataclass(frozen=True)
class FrozenProtocol:
    directory: Path
    raw: dict[str, Any]
    wells: dict[str, WellSpec]
    folds: dict[int, FoldSpec]

    @property
    def declared_data_root(self) -> Path:
        """Original absolute data path recorded when the protocol was frozen."""

        return Path(self.raw["dataset"]["root"]).expanduser()

    def _is_data_root(self, candidate: Path) -> bool:
        five_curve_specs = (
            spec for spec in self.wells.values() if spec.role != "four_curve_sensitivity"
        )
        sentinel = next(five_curve_specs, None)
        return (
            sentinel is not None
            and candidate.is_dir()
            and (candidate / sentinel.source_relative_path).is_file()
        )

    @property
    def data_root(self) -> Path:
        """Resolve the frozen dataset after the project has been moved as one folder."""

        env_path = os.environ.get("GAGRU_DATA_ROOT")
        if env_path:
            explicit = Path(env_path).expanduser().resolve()
            if self._is_data_root(explicit):
                return explicit
            raise ProtocolError(
                "GAGRU_DATA_ROOT does not contain the frozen well-file layout: "
                f"{explicit}"
            )

        declared = self.declared_data_root.resolve()
        dataset_name = self.declared_data_root.name
        project_dir = self.directory.parent
        anchors = (project_dir, project_dir.parent, project_dir.parent.parent)
        candidates = [declared]
        for anchor in anchors:
            candidates.extend(
                [
                    anchor / "测井文件集" / dataset_name,
                    anchor / dataset_name,
                ]
            )

        checked: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            key = str(resolved).casefold()
            if key in seen:
                continue
            seen.add(key)
            checked.append(str(resolved))
            if self._is_data_root(resolved):
                return resolved
        raise ProtocolError(
            "Frozen dataset root was not found. Set GAGRU_DATA_ROOT to the folder "
            "that contains 01_模型开发与井级验证. Checked: " + "; ".join(checked)
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(item["name"] for item in self.raw["features"])

    @property
    def resistivity_names(self) -> tuple[str, ...]:
        return tuple(
            item["name"]
            for item in self.raw["features"]
            if item["transform"] == "log10_then_standardize"
        )

    @property
    def class_names(self) -> dict[int, str]:
        return {item["class_id"]: item["class_name"] for item in self.raw["labels"]}

    @property
    def legacy_class_ids(self) -> dict[int, int]:
        return {item["class_id"]: item["legacy_class_id"] for item in self.raw["labels"]}

    @property
    def locked_external_wells(self) -> frozenset[str]:
        return frozenset(self.raw["roles"]["locked_external"])

    def wells_for_role(self, role: str) -> tuple[str, ...]:
        if role not in self.raw["roles"]:
            raise ProtocolError(f"Unknown protocol role: {role}")
        return tuple(self.raw["roles"][role])

    def well(self, well_id: str) -> WellSpec:
        try:
            return self.wells[well_id]
        except KeyError as exc:
            raise ProtocolError(f"Well is not declared in the frozen protocol: {well_id}") from exc

    def fold(self, fold_number: int) -> FoldSpec:
        try:
            return self.folds[fold_number]
        except KeyError as exc:
            raise ProtocolError(f"Unknown outer fold: {fold_number}") from exc

    def assert_access_allowed(
        self,
        well_ids: Iterable[str],
        *,
        allow_locked_external: bool = False,
    ) -> None:
        requested = set(well_ids)
        unknown = requested.difference(self.wells)
        if unknown:
            raise ProtocolError(f"Unknown wells requested: {sorted(unknown)}")
        locked = requested.intersection(self.locked_external_wells)
        if locked and not allow_locked_external:
            names = ", ".join(sorted(locked))
            raise ExternalTestLockedError(
                f"Locked external wells cannot be read during development: {names}. "
                "The final external-test runner must explicitly unlock and write the access log."
            )


def load_frozen_protocol(protocol_dir: str | Path | None = None) -> FrozenProtocol:
    directory = resolve_protocol_dir(protocol_dir)
    raw = json.loads((directory / PROTOCOL_JSON).read_text(encoding="utf-8"))
    if raw.get("protocol", {}).get("status") != "FROZEN_FOR_DEVELOPMENT":
        raise ProtocolError("Protocol status is not FROZEN_FOR_DEVELOPMENT")

    feature_names = [item["name"] for item in raw.get("features", [])]
    if len(feature_names) != 5 or len(set(feature_names)) != 5:
        raise ProtocolError("Frozen protocol must declare five unique model features")
    labels = raw.get("labels", [])
    if sorted(item["class_id"] for item in labels) != list(range(10)):
        raise ProtocolError("Frozen protocol class_id values must be exactly 0-9")
    if len({item["class_name"] for item in labels}) != 10:
        raise ProtocolError("Frozen protocol class names must be unique")
    if len({item["legacy_class_id"] for item in labels}) != 10:
        raise ProtocolError("Frozen protocol legacy_class_id values must be unique")

    well_rows = _read_csv(directory / WELL_MANIFEST)
    wells: dict[str, WellSpec] = {}
    for row in well_rows:
        well_id = row["well_id"]
        if well_id in wells:
            raise ProtocolError(f"Duplicate well in manifest: {well_id}")
        wells[well_id] = WellSpec(
            well_id=well_id,
            role=row["protocol_role"],
            source_relative_path=row["source_relative_path"],
            rows_5curve=int(row["rows_5curve"]),
            continuous_segments=int(row["continuous_segments"]),
            boundary_rows=int(row["boundary_rows_0_25m"]),
            class_ids_present=tuple(int(value) for value in _split_wells(row["class_ids_present"])),
            outer_fold=int(row["outer_fold"]) if row["outer_fold"] else None,
            external_locked=row["external_locked"].strip().lower() == "true",
            reason=row["reason"],
        )

    configured_wells = {
        well_id
        for role_wells in raw["roles"].values()
        for well_id in role_wells
    }
    if set(wells) != configured_wells:
        raise ProtocolError("Well manifest and JSON role lists do not contain the same wells")
    for role, well_ids in raw["roles"].items():
        for well_id in well_ids:
            if wells[well_id].role != role:
                raise ProtocolError(
                    f"Role mismatch for {well_id}: JSON={role}, manifest={wells[well_id].role}"
                )
    locked = set(raw["roles"]["locked_external"])
    manifest_locked = {well_id for well_id, spec in wells.items() if spec.external_locked}
    if locked != manifest_locked:
        raise ProtocolError("External lock flags disagree between JSON and well manifest")

    folds: dict[int, FoldSpec] = {}
    for row in _read_csv(directory / FOLD_MANIFEST):
        if row["outer_fold"] == "final":
            continue
        fold_number = int(row["outer_fold"])
        folds[fold_number] = FoldSpec(
            fold=fold_number,
            outer_test_well=row["outer_test_well"],
            inner_validation_wells=_split_wells(row["inner_validation_wells"]),
            inner_training_wells=_split_wells(row["inner_training_wells"]),
            outer_training_wells=_split_wells(row["outer_training_wells"]),
        )

    if set(folds) != set(range(1, 8)):
        raise ProtocolError(f"Expected outer folds 1-7, found {sorted(folds)}")
    return FrozenProtocol(directory=directory, raw=raw, wells=wells, folds=folds)
