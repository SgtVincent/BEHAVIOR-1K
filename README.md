<h1 align="center">BEHAVIOR-1K</h1>

![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)

**BEHAVIOR-1K** is a comprehensive simulation benchmark for testing embodied AI agents on 1,000 everyday household activities. This monolithic repository provides everything needed to train and evaluate agents on human-centered tasks like cleaning, cooking, and organizing — activities selected from real human time-use surveys and preference studies.

***Check out our [main website](https://behavior.stanford.edu/) for more details!***

# 🛠️ Installation

BEHAVIOR-1K provides an installation script that handles all dependencies and components. The script supports modular installation, allowing you to install only the components you need.

Please check out [Installation Guide](https://behavior.stanford.edu/getting_started/installation.html) for more details!

# 📚 Repo-local Guides

除了公开网站文档外，仓库内还维护了一组便于直接跟代码对照的说明文档，放在 [`BEHAVIOR-1K/.trae/documents/`](.trae/documents/)：

| 你要做什么 | 直接看这里 |
| --- | --- |
| 先了解仓库模块与主要入口 | [`project_overview.md`](.trae/documents/project_overview.md) |
| 跑 behavior / simulator 侧 eval、查 runtime flags | [`behavior_eval_runtime_guide.md`](.trae/documents/behavior_eval_runtime_guide.md) |
| 看 skill / segment eval 的使用方式 | [`skill_eval_user_guide.md`](.trae/documents/skill_eval_user_guide.md) |
| 看 RLinf-style 吞吐优化与默认推荐组合 | [`rlinf-style-eval-optimization.md`](.trae/documents/rlinf-style-eval-optimization.md) |

如果你只想先用起来，建议按这个顺序：

1. 先按本 README 和官方 Installation Guide 完成安装；
2. 跑仿真 / eval 时先看 [`behavior_eval_runtime_guide.md`](.trae/documents/behavior_eval_runtime_guide.md)；
3. 做 skill / segment eval 时再看 [`skill_eval_user_guide.md`](.trae/documents/skill_eval_user_guide.md)；
4. 要优化吞吐、调 `partial_scene_load` / `skip_intermediate_obs_in_chunk` 时，再看 [`rlinf-style-eval-optimization.md`](.trae/documents/rlinf-style-eval-optimization.md)。

## 📄 Citation

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```
