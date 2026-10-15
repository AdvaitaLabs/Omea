"""Metrics use all raw attempts; missing geometry checks cannot count as passes."""
import json
import math
from benchmark_io import CODE_ROOT

REQUIRED_CHECKS = frozenset(json.loads((CODE_ROOT/'configs/geometry_schema.json').read_text())['required_checks'])


def geometry_result(score):
    checks = score.get('posebusters_checks', {})
    missing = sorted(REQUIRED_CHECKS-set(checks))
    passed = bool(score.get('status') == 'scored' and not missing and
                  checks and all(bool(v) for v in checks.values()) and not score.get('posebusters_error'))
    return passed, missing


def outcomes(attempt, score, reference):
    geometry, missing = geometry_result(score)
    scored = score.get('status') == 'scored' and all(
        isinstance(score.get(key), (int, float)) and math.isfinite(score[key])
        for key in ('vina_kcal_mol', 'cnn_affinity_pK'))
    valid = bool(attempt.get('strict_success') and geometry and scored)
    new = valid and bool(attempt.get('new_design'))
    ref_scored = reference.get('status') == 'scored'
    dv = score['vina_kcal_mol']-reference['vina_kcal_mol'] if scored and ref_scored else None
    dc = score['cnn_affinity_pK']-reference['cnn_affinity_pK'] if scored and ref_scored else None
    dual = bool(new and dv is not None and dv < 0 and dc > 0)
    return dict(valid_with_geometry=valid, valid_new=new, dual_improved=dual,
                charge_matched_dual=dual and bool(attempt.get('charge_match')),
                delta_vina_kcal_mol=dv, delta_cnn_affinity_pK=dc,
                missing_geometry_checks=missing)
