"""Predeclared full-cohort CNN ablations; affinity evaluation is separated."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/direct_guidance_selected_x0p02_h0p6_scale30_v1'
CHECKS = ROOT / 'results/cnn_clean_ablation_checks_v1'
RUN = ROOT / 'results/cnn_clean_ablations_scale30_v1'
COMMON = dict(lambda_x=.02, lambda_h=.6, last_fraction=1., interval=1,
              schedule='flat', temperature=.2, objective='cnn_affinity',
              late_steps=100, taper_end=.6, geometry_weight=0.,
              geometry_gradient_scale=1., bond_relative_tolerance=.10,
              angle_tolerance_degrees=15.)
BASELINE = dict(COMMON, name='x0p02_h0p6')
SETTINGS = {
    'cnn_element_late_taper': dict(COMMON, name='cnn_element_late_taper', taper_end=.2),
    'cnn_local_geometry_guard': dict(COMMON, name='cnn_local_geometry_guard', geometry_weight=.1),
}
ARMS = ('zero', 'x0p02_h0p6', *SETTINGS)
NEW_SOURCES = ('cnn_clean_settings.py', 'cnn_local_geometry.py', 'autofragdiff_cnn_clean_hook.py',
               'score_cnn_clean.py', 'summarize_cnn_clean_full.py', 'check_cnn_clean_ablations.py',
               'run_cnn_clean_ablations.py')


def element_strength(config, step):
    start = config['late_steps']
    if step >= start: return config['lambda_h']
    return config['taper_end'] + (config['lambda_h']-config['taper_end'])*step/start
