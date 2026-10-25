"""Frozen C/D continuation: CNN-only generation, both metrics always evaluated."""
from cnn_clean_settings import ROOT, COMMON

SOURCE = ROOT / 'results/cnn_clean_dual_evaluation_v1'
PARENT = ROOT / 'results/cnn_clean_ablations_scale30_v1'
CHECKS = ROOT / 'results/cnn_geometry_followup_checks_v1'
RUN = ROOT / 'results/cnn_geometry_followup_scale30_v1'
BASELINE = dict(COMMON, name='cnn_local_geometry_guard', geometry_weight=.1,
                geometry_projection=False, clash_radius_fraction=.65,
                projection_iterations=32, backtracking_steps=16)
SETTINGS = {
    'cnn_geometry_weight_0p3': dict(BASELINE, name='cnn_geometry_weight_0p3', geometry_weight=.3),
    'cnn_geometry_projected': dict(BASELINE, name='cnn_geometry_projected', geometry_projection=True),
}
ARMS = ('zero', BASELINE['name'], *SETTINGS)
NEW_SOURCES = ('cnn_geometry_followup_settings.py', 'cnn_geometry_projection.py',
               'autofragdiff_cnn_geometry_hook.py', 'check_cnn_geometry_followup.py',
               'summarize_cnn_geometry_followup.py', 'run_cnn_geometry_followup.py')
