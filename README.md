# Adversarial 2048

一个敌对版 2048 AI 实验平台。玩家尽量合成更大的数字，系统可以选择更坏的出块方式来压制玩家，用来比较普通策略、轻量 Q-learning、深度 DQN 和后续搜索敌人的表现。

当前项目已经包含标准 2048 环境、CLI、Tkinter GUI、CSV/JSONL 记录、Q-learning 训练、DQN 训练、训练成果比较、模型合并、自动短轮调参和结果管理。

## 运行

默认启动 GUI：

```powershell
python src/main.py
```

使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe src/main.py
```

常用 CLI：

```powershell
python src/main.py play --enemy random
python src/main.py auto --player heuristic --enemy random --episodes 100
python src/main.py auto --player q_ai --enemy q_enemy --episodes 100
python src/main.py train-player --episodes 300 --enemy random
python src/main.py train-enemy --episodes 300 --player heuristic
python src/main.py train-player-dqn --episodes 300 --enemy random
python src/main.py train-enemy-dqn --episodes 300 --player heuristic
python src/main.py training-list
python src/main.py training-compare --a models/q_learning/player/.../player_q_model.json --b models/q_learning/player/.../player_q_model.json --episodes 30
python src/main.py training-merge --a models/q_learning/player/.../player_q_model.json --b models/q_learning/player/.../player_q_model.json
python src/main.py training-tune --target player --algorithm q_learning
```

## GUI

GUI 包含四个工作区：

- `对局`：手动玩、AI 单步、自动播放，并切换当前敌人。
- `单项评估平台`：选择自动玩家、自动敌人或某次训练成果运行评估。
- `模型训练`：训练玩家 AI 或敌对 AI，支持参考模型、继续训练、停止并保存未完成进度。
- `训练评估平台`：比较两次训练成果、合并同类型模型、运行调参候选。

主窗口还提供结果管理和系统日志清理。`latest/` 模型是当前默认加载入口，删除或替换前需要明确确认。

## 策略类型

玩家类型：

- `human`：人类玩家，用于 `play` 和 GUI。
- `random`：随机合法动作。
- `heuristic`：一层启发式搜索，按空格、合并机会、最大块、角落、单调性和得分增量评分。
- `q_ai`：轻量 Q-learning 玩家，默认加载 `models/q_learning/player/latest/player_q_model.json`。
- `dqn_player`：深度 DQN 玩家，默认加载 `models/dqn/player/latest/player_dqn_model.pt`。

敌人类型：

- `random`：标准 2048 出块，90% 生成 2，10% 生成 4。
- `greedy`：枚举所有空格和 2/4，选择当前评分最差的生成方式。
- `q_enemy`：轻量 Q-learning 敌人，默认加载 `models/q_learning/enemy/latest/enemy_q_model.json`。
- `dqn_enemy`：深度 DQN 敌人，默认加载 `models/dqn/enemy/latest/enemy_dqn_model.pt`。

历史别名已经移除。只使用上面的正式名称，旧名称会直接报错，避免脚本和 GUI 选项继续漂移。

## 配置

可调默认值集中在 `configs/`：

```text
configs/
├── paths.yaml
├── evaluate/default.yaml
├── ui/default.yaml
└── train/
    ├── player_q.yaml
    ├── enemy_q.yaml
    ├── player_dqn.yaml
    ├── enemy_dqn.yaml
    └── tuning.yaml
```

关键路径字段：

- `runs_directory`：某类训练成果的历史目录。
- `latest_output_directory`：某类训练成果发布为默认模型的 `latest/` 目录。
- `outputs.experiments_directory`：正式评估 CSV 的输出根目录。
- `logs.directory`：系统日志根目录。
- `logs.errors_directory`：错误日志目录。

CLI 参数可以覆盖 YAML 默认值。例如 `--episodes`、`--learning-rate`、`--gamma` 和 `--output` 只影响当次运行。`--output` 是显式输出路径，不是 YAML 配置字段。

## 训练与评估产物

正式评估默认写入：

```text
outputs/experiments/<timestamp>/<player>_vs_<enemy>.csv
outputs/experiments/<timestamp>/log.jsonl
```

训练默认写入：

```text
models/q_learning/player/<timestamp>/player_q_model.json
models/q_learning/player/<timestamp>/info.json
models/q_learning/player/<timestamp>/log.jsonl
models/q_learning/player/latest/player_q_model.json
```

DQN 训练还会按稳定性配置保存旁路 checkpoint：

```text
player_dqn_model_best.pt
player_dqn_model_checkpoint.pt
```

继续训练只接受同类型未完成训练目录。新的继续训练保存成功后，会删除旧的未完成目录；完整参考模型只作为初始化权重来源，不会被自动删除。

## PyTorch / GPU

默认依赖使用 NVIDIA GPU 训练版 PyTorch（CUDA 12.8）：

```powershell
python -m pip install -r requirements.txt
```

CPU-only 环境使用：

```powershell
python -m pip install -r requirements-cpu.txt
```

程序会自动选择 `cuda` 或 `cpu`。如果 DQN 相关命令提示缺少 PyTorch，优先使用项目虚拟环境重新验证：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## 测试

```powershell
python -m unittest discover -s tests
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

系统 Python 可以覆盖标准库和非 Torch 路径；项目虚拟环境用于完整 DQN 回归。

## 开发路线

项目原则是先让平台可靠运行，再逐步让 AI 变强：

1. 稳定游戏环境、记录系统和 GUI 工作流。
2. 用随机、启发式、Q-learning 和 DQN 建立可比较基线。
3. 增加更强的敌对搜索，例如 minimax depth=1/2。
4. 增加训练曲线、最大块分布和对战矩阵等报告能力。
5. 做玩家和敌人轮流训练，比较不同版本的泛化能力。

阶段验收不靠感觉，而看固定 seed、多局均值、最大块分布、达到指定 tile 的概率和训练成果之间的横向比较。
