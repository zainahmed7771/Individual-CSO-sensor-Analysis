from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from cso_spatial_drivers.config import load_config, validate_input_contracts, validate_input_paths
from cso_spatial_drivers.workflow import STAGES, run_pipeline


def test_demo_configuration_has_complete_inputs():
    config = load_config("config/demo.yaml")
    assert config.profile == "teaching_demo"
    assert validate_input_paths(config) == []
    assert validate_input_contracts(config) == []
    assert config.analysis["tail_rule"] == "strictly_greater_than"


def test_end_to_end_demo_builds_all_contract_outputs(tmp_path):
    config = replace(load_config("config/demo.yaml"), paths={**load_config("config/demo.yaml").paths, "output_root": tmp_path / "run"})
    manifest = run_pipeline(config)
    assert [row["stage"] for row in manifest["stages"]] == list(STAGES)
    master = pd.read_csv(tmp_path / "run/03_master/scientific_master.csv")
    assert len(master) == 18
    assert not master.duplicated(["company", "uwwCode"]).any()
    assert (master["spill_events_per_monitoring_year"] > 0).all()
    scorecard = pd.read_csv(tmp_path / "run/05_machine_learning/model_scorecard.csv")
    assert set(scorecard["outcome"]) == {"beta", "lambda", "duration", "frequency"}
    assert (tmp_path / "run/run_manifest.json").exists()
    assert (tmp_path / "run/06_report/RUN_SUMMARY.md").exists()
    with pytest.raises(FileExistsError, match="Use --force"):
        run_pipeline(config)
    replaced = run_pipeline(config, force=True)
    assert replaced["overwrite_requested"] is True


def test_scientific_threshold_cannot_be_changed():
    config = load_config("config/demo.yaml")
    invalid = replace(config, analysis={**config.analysis, "tail_threshold_minutes": 239})
    assert any("must remain 240" in problem for problem in validate_input_paths(invalid))


def test_assignments_require_explicit_evidence(tmp_path):
    original = load_config("config/demo.yaml")
    assignments = pd.read_csv(original.paths["assignments"])
    assignments.loc[0, "assignment_evidence"] = ""
    bad_path = tmp_path / "assignments.csv"
    assignments.to_csv(bad_path, index=False)
    config = replace(original, paths={**original.paths, "assignments": bad_path, "output_root": tmp_path / "run"})
    with pytest.raises(ValueError, match="requires assignment_evidence"):
        run_pipeline(config, through="master")
