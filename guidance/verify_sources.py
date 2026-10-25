"""Check the exported historical guidance sources without importing ML packages."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(ok,message):
    if not ok:raise SystemExit('FAIL: '+message)

meta=json.loads((ROOT/'frozen_sources.json').read_text())
expected={}
for key in ['frozen_scripts_sha256','separately_frozen_scripts_sha256','supporting_scripts_sha256']:
    expected.update(meta[key])
actual={p.name for p in (ROOT/'scripts').glob('*.py')}
require(actual==set(expected),'脚本清单变化')
require(len(meta['frozen_scripts_sha256'])==45 and len(expected)==47,'冻结脚本数量变化')
require(not actual.intersection(meta['excluded_scripts']),'混入后续实验代码')
for name,value in expected.items():
    path=ROOT/'scripts'/name
    require(sha(path)==value,'历史源码哈希不符: '+name)
    ast.parse(path.read_text(),filename=name)
    for module in meta['local_imports'][name]:
        require((ROOT/'scripts'/(module+'.py')).exists(),'缺少依赖: '+module)
require(sha(ROOT/'results/selection_rules_v3/geometry_schema.json')==meta['geometry_schema_sha256'],
        '历史几何规则变化')
require(sha(ROOT.parent/'configs/guidance_configurations.json')==meta['configuration_sha256'],
        '历史配置变化')
result=json.loads((ROOT.parent/'docs/guidance_validation.json').read_text())
require(result['audit']['passed'] and result['audit']['native_final_reconstruction'],
        '原生最终重建版本验收不符')
require(result['selection']['uses_external_score'] is False,'选择规则使用了外部评估分数')
for name,count in {'zero':360,'cnn_local_geometry_guard':492,
                   'cnn_geometry_weight_0p3':494,'cnn_geometry_projected':495}.items():
    row=result['all_30'][name]
    require(row['attempts']==3000 and row['dual_success']==count,'周报统计不符: '+name)
print('PASS: 47 guidance scripts; 45 frozen sources + 1 separately frozen evaluator; native-final 16.50% version.')
