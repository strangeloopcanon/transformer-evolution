from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import yaml
from jsonschema import Draft202012Validator

from .errors import DSLValidationError
from .models import DSLConfig
from .validators import run_additional_checks
from .schema import DSL_JSON_SCHEMA


def dump_schema(path: Path) -> None:
    path.write_text(json.dumps(DSL_JSON_SCHEMA, indent=2))


def _validate_schema(cfg_dict: dict) -> None:
    validator = Draft202012Validator(DSL_JSON_SCHEMA)
    errors = sorted(validator.iter_errors(cfg_dict), key=lambda e: e.path)
    if errors:
        msg = "\n".join(
            [
                f"{list(e.path)}: {e.message}" if e.path else e.message
                for e in errors
            ]
        )
        raise DSLValidationError(msg)


def load_validate_yaml(path: Path) -> DSLConfig:
    cfg_dict = yaml.safe_load(path.read_text())
    _validate_schema(cfg_dict)
    try:
        cfg = DSLConfig.model_validate(cfg_dict)
    except Exception as exc:  # pydantic ValidationError
        raise DSLValidationError(str(exc)) from exc
    run_additional_checks(cfg)
    return cfg
