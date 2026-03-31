# TC-ADV

`TC-ADV` 是论文第二个核心工作的科研工程仓库，定位为 `LMCA-TIC` 生成器的对抗验证扩展层。

最小工作流：

```bash
python3 -m pip install -e .
tc-adv prepare-data --events /path/to/events.csv --entities /path/to/entities.csv --output-root data/local/icews18
tc-adv train --config configs/experiments/smoke.yaml
tc-adv evaluate --config configs/experiments/smoke.yaml
tc-adv run-suite --config configs/experiments/smoke.yaml
tc-adv export-code --output outputs/code_bundle.md
```
