# Adversarial_2048 当前项目结构

本文档记录当前真实文件结构，并标明实现状态，方便后续继续开发。  
状态含义：

- `[已实现]`：当前已有可运行代码或有效文档。
- `[占位]`：目录或包已建立，但还没有实质功能。
- `[未实现]`：计划中需要补充，但当前还不存在或尚未开发。

运行入口：

```powershell
python src/main.py
```

测试入口：

```powershell
python -m unittest discover -s tests
```

说明：`__pycache__/`、`.pyc`、`outputs/experiments/` 中生成的 CSV/JSONL、`logs/system/` 与 `logs/errors/` 中生成的 JSONL、`models/` 中训练生成的模型文件属于运行产物，不作为源码结构记录。

---

## 当前目录树

```text
adversarial_2048/
│
├── README.md                         # [已实现] 项目说明、运行方式、模式说明
├── requirements.txt                  # [已实现] 默认 GPU 训练依赖，PyTorch CUDA 12.8 + numpy
├── requirements-gpu-cu128.txt        # [已实现] GPU 版 PyTorch CUDA 12.8 依赖
├── requirements-cpu.txt              # [已实现] CPU-only 备选依赖
├── .gitignore                        # [已实现] 忽略缓存、日志、输出、模型等运行产物
├── File.md                           # [已实现] 当前文件结构与实现状态说明
│
├── src/                              # [已实现] 源码根目录
│   ├── __init__.py                   # [已实现] src 包标记
│   ├── main.py                       # [已实现] 纯程序入口；解析参数后分发到 CLI 命令
│   │
│   ├── cli/                          # [已实现] 命令行入口层
│   │   ├── __init__.py               # [已实现] 导出 build_parser / dispatch
│   │   ├── parser.py                 # [已实现] argparse 命令和参数定义
│   │   ├── commands.py               # [已实现] CLI 命令执行、终端游玩、训练/评估分发
│   │   └── formatters.py             # [已实现] CLI JSON 输出格式化
│   │
│   ├── config/                       # [已实现] 配置加载包
│   │   └── __init__.py               # [已实现] YAML/简易 YAML 配置加载器，提供路径、训练、评估、UI 默认值
│   │
│   ├── domain/                       # [已实现] 2048 领域层：规则、策略、模型、训练、评估和结果管理
│   │   ├── __init__.py               # [已实现] domain 包标记
│   │   ├── game/                     # [已实现] 2048 游戏核心逻辑
│   │   │   ├── __init__.py           # [已实现] 导出 GameEnv / GameState
│   │   │   ├── board.py              # [已实现] 棋盘创建、复制、空格、最大块、放置方块、格式化
│   │   │   ├── constants.py          # [已实现] 棋盘大小、动作常量、默认出块概率
│   │   │   ├── env.py                # [已实现] GameEnv：reset、step、spawn、legal actions、snapshot
│   │   │   ├── rules.py              # [已实现] 移动合并规则、合法动作、结束判断、启发式评分
│   │   │   └── state.py              # [已实现] GameState：棋盘、分数、步数、结束状态
│   │   ├── players/                  # [已实现] 玩家策略
│   │   │   ├── __init__.py           # [已实现] create_player 工厂
│   │   │   ├── base_player.py        # [已实现] BasePlayer 接口
│   │   │   ├── human_player.py       # [已实现] 命令行人类玩家输入
│   │   │   ├── random_player.py      # [已实现] 随机合法动作玩家
│   │   │   ├── heuristic_player.py   # [已实现] 启发式玩家
│   │   │   ├── q_player.py           # [已实现] 轻量 Q-learning 玩家
│   │   │   └── dqn_player.py         # [已实现] PyTorch 深度 DQN 玩家
│   │   ├── enemies/                  # [已实现] 敌对出块策略
│   │   │   ├── __init__.py           # [已实现] create_enemy 工厂
│   │   │   ├── base_enemy.py         # [已实现] BaseEnemy 接口
│   │   │   ├── random_enemy.py       # [已实现] RandomEnemy 标准随机敌人
│   │   │   ├── greedy_enemy.py       # [已实现] 贪心敌人，枚举空格和 2/4 选择最坏局面
│   │   │   ├── q_enemy.py            # [已实现] 轻量 Q-learning 敌人
│   │   │   └── dqn_enemy.py          # [已实现] PyTorch 深度 DQN 敌人
│   │   ├── models/                   # [已实现/待扩展] 模型定义和模型工具
│   │   │   ├── __init__.py           # [已实现] 导出 Q-learning 模型和敌人动作映射
│   │   │   ├── q_learning/           # [已实现] 轻量 Q-learning 模型包
│   │   │   │   ├── __init__.py       # [已实现] 导出玩家/敌人 Q 模型
│   │   │   │   ├── player.py         # [已实现] 纯标准库玩家线性 Q 模型
│   │   │   │   └── enemy.py          # [已实现] 纯标准库敌人线性 Q 模型，32 个出块动作
│   │   │   ├── torch_utils.py        # [已实现] PyTorch 检测、CUDA/CPU 设备选择
│   │   │   └── dqn_network.py        # [已实现] PyTorch MLP DQN 网络
│   │   ├── train/                    # [已实现/待扩展] 训练脚本目录
│   │   │   ├── __init__.py           # [已实现] 导出 Q/DQN 训练入口，DQN 按需懒加载
│   │   │   ├── artifacts.py          # [已实现] 训练产物命名、索引、元数据保存与查询
│   │   │   ├── merge.py              # [已实现] 兼容训练产物合并与发布 latest
│   │   │   ├── tuning.py             # [已实现] 自动短轮调参与候选结果排序
│   │   │   ├── q_learning/           # [已实现] Q-learning 训练子包
│   │   │   │   ├── __init__.py       # [已实现] 导出 Q-learning 训练入口和奖励函数
│   │   │   │   ├── player.py         # [已实现] Q-learning 玩家训练循环
│   │   │   │   └── enemy.py          # [已实现] Q-learning 敌人训练循环
│   │   │   └── dqn/                  # [已实现] DQN 训练子包
│   │   │       ├── __init__.py       # [已实现] 导出 DQN 训练入口，按需懒加载 Torch
│   │   │       ├── checkpoints.py    # [已实现] DQN checkpoint 和 state_dict 共享工具
│   │   │       ├── replay_buffer.py  # [已实现] DQN 经验回放池
│   │   │       ├── stability.py      # [已实现] DQN 稳定性控制、回滚、学习率调整配置
│   │   │       ├── player.py         # [已实现] PyTorch DQN 玩家训练循环
│   │   │       └── enemy.py          # [已实现] PyTorch DQN 敌人训练循环
│   │   ├── evaluation/               # [已实现] 自动对局与实验运行
│   │   │   ├── __init__.py           # [已实现] 导出 run_episode / run_experiment
│   │   │   ├── compare.py            # [已实现] 训练产物识别、加载与横向评估比较
│   │   │   ├── run_episode.py        # [已实现] 单局自动对战
│   │   │   └── experiment.py         # [已实现] 多局实验、CSV 输出、GUI/CLI 共享进度回调
│   │   └── results/                  # [已实现] 训练、评估和日志产物的统一管理服务
│   │       ├── __init__.py           # [已实现] 导出结果列表、删除、日志压缩和 latest 发布接口
│   │       └── management.py         # [已实现] 安全列出/删除产物、清理日志、设为 latest
│   │
│   ├── workflows/                    # [已实现] UI 到训练/评估领域层之间的纯工作流适配工具
│   │   ├── __init__.py               # [已实现] 工作流适配包
│   │   ├── training.py               # [已实现] 训练参考/继续训练产物、参考类型 key 和输出路径解析
│   │   └── evaluation.py             # [已实现] 单项评估选项、请求对象和输出路径解析
│   │
│   ├── ui/                           # [已实现] Tkinter 图形界面
│   │   ├── __init__.py               # [已实现] 导出 App / BoardView / run_gui
│   │   ├── app.py                    # [已实现] 主窗口装配、应用状态、面板切换、键盘绑定
│   │   ├── components/               # [已实现] 可复用 UI 组件，按控件类型拆分
│   │   │   ├── __init__.py           # [已实现] 包级导出棋盘、按钮、输入、消息区和栅格控件选项
│   │   │   ├── board_view.py         # [已实现] 棋盘 Canvas 绘制与方块移动动画
│   │   │   ├── buttons.py            # [已实现] 操作按钮构造和按钮视觉状态切换
│   │   │   ├── inputs.py             # [已实现] 下拉框、步进器、文本输入框和 GRID_CONTROL_OPTIONS
│   │   │   ├── messages.py           # [已实现] 状态提示区和结果文本区构造
│   │   │   └── controls.py           # [已实现] 兼容导出层，转发到 buttons/inputs/messages
│   │   ├── panels/                   # [已实现] 主界面功能面板
│   │   │   ├── __init__.py           # [已实现] 导出功能面板构造器
│   │   │   ├── game.py               # [已实现] 对局控制面板，使用区域栅格布局
│   │   │   ├── training.py           # [已实现] 模型训练面板，使用区域栅格布局并复用训练后端
│   │   │   ├── evaluation.py         # [已实现] 单项评估平台面板，使用区域栅格布局
│   │   │   └── training_platform.py  # [已实现] 训练平台：产物列表、合并、自动调参入口，使用区域栅格布局
│   │   ├── settings/                 # [已实现] UI 主题、固定尺寸和选项配置
│   │   │   ├── __init__.py           # [已实现] UI 配置包
│   │   │   ├── layout/               # [已实现] UI 布局基础尺寸和区域布局模板
│   │   │   │   ├── __init__.py       # [已实现] 导出布局基础尺寸和栅格 helper
│   │   │   │   ├── base.py           # [已实现] 全局窗口、棋盘、侧栏、表单尺寸和锁定尺寸工具
│   │   │   │   └── grid.py           # [已实现] 20x9 区域栅格、坐标/跨度计算和统一面板创建
│   │   │   ├── options.py            # [已实现] GUI 文案标签与内部类型映射
│   │   │   └── theme.py              # [已实现] GUI 主题、颜色、字体、按钮样式、高 DPI 设置
│   │   └── windows/                  # [已实现] 独立弹窗
│   │       ├── __init__.py           # [已实现] 导出窗口构造器
│   │       └── result_manager.py     # [已实现] 训练模型和评估结果管理窗口
│   │
│   └── utils/                        # [已实现] 通用工具
│       ├── __init__.py               # [已实现] 导出 recorder / seed / training_log
│       ├── recorder.py               # [已实现] EpisodeRecord 和 ExperimentRecorder，保存 CSV
│       ├── seed.py                   # [已实现] 随机种子工具
│       ├── serialization.py          # [已实现] JSON/Path 清洗和 checkpoint 元数据转换
│       └── training_log.py           # [已实现] 单次训练/评估成果日志与错误日志 JSONL 写入
│
├── tests/                            # [已实现] 标准库 unittest 测试
│   ├── __init__.py                   # [已实现] tests 包标记
│   ├── _path.py                      # [已实现] 测试时加入 src 路径
│   ├── test_rules.py                 # [已实现] 移动合并、非法移动、结束判断
│   ├── test_players_enemies.py       # [已实现] 玩家动作合法性、敌人出块合法性
│   ├── test_end_to_end.py            # [已实现] 自动对局、实验 CSV、无参数 GUI 入口解析
│   ├── test_q_player.py              # [已实现] Q 模型保存加载、AI 玩家动作、训练写出模型
│   ├── test_q_enemy.py               # [已实现] 敌人 Q 模型保存加载、敌对 AI 动作、训练写出模型
│   ├── test_stability.py             # [已实现] DQN 稳定性配置与控制器测试
│   ├── test_result_management.py     # [已实现] 结果管理、安全删除、日志压缩和 latest 发布测试
│   └── test_training_platform.py     # [已实现] 训练产物、合并、比较、自动调参测试
│
├── docs/                             # [已实现/待扩展] 项目文档
│   └── project_plan.md               # [已实现] v1 范围与后续阶段
│
├── configs/                          # [已实现] YAML 配置目录
│   ├── .gitkeep                      # [已实现] 保留目录
│   ├── paths.yaml                    # [已实现] 模型路径、日志目录等共享路径配置
│   ├── evaluate/                     # [已实现] 批量评估配置
│   │   └── default.yaml              # [已实现] 默认玩家、敌人、局数、seed、输出与 max_steps
│   ├── ui/                           # [已实现] GUI 默认配置
│   │   └── default.yaml              # [已实现] 初始玩家、敌人、实验区和训练区默认值
│   └── train/                        # [已实现] 训练配置
│       ├── player_q.yaml             # [已实现] Q-learning 玩家训练默认参数
│       ├── enemy_q.yaml              # [已实现] Q-learning 敌人训练默认参数
│       ├── player_dqn.yaml           # [已实现] DQN 玩家训练默认参数
│       └── enemy_dqn.yaml            # [已实现] DQN 敌人训练默认参数
│
├── logs/                             # [已实现] 系统和错误日志目录
│   ├── .gitkeep                      # [已实现] 保留目录；实际日志被 gitignore 忽略
│   ├── system/                       # [已实现] 系统 JSONL 日志
│   │   └── .gitkeep                  # [已实现] 保留目录；实际系统日志被 gitignore 忽略
│   └── errors/                       # [已实现] 异常 JSONL 日志
│       └── .gitkeep                  # [已实现] 保留目录；实际 log.jsonl 被 gitignore 忽略
│
├── outputs/                          # [已实现] 正式输出目录
│   ├── .gitkeep                      # [已实现] 保留目录
│   ├── experiments/                  # [已实现] 正式实验时间戳目录，内含 CSV 与 log.jsonl
│   │   └── .gitkeep                  # [已实现] 保留目录；实际评估产物被 gitignore 忽略
│   └── dqn_smoke/                    # [运行产物] DQN 冒烟/验证训练输出
│
└── models/                           # [已实现] 训练生成的模型产物目录
    ├── .gitkeep                      # [已实现] 保留目录
    ├── dqn/                          # [已实现] PyTorch DQN 权重
    │   ├── .gitkeep                  # [已实现] 保留目录
    │   ├── player/                   # [已实现] 玩家 DQN 模型
    │   │   └── .gitkeep              # [已实现] 保留目录；实际 .pt 被 gitignore 忽略
    │   └── enemy/                    # [已实现] 敌人 DQN 模型
    │       └── .gitkeep              # [已实现] 保留目录；实际 .pt 被 gitignore 忽略
    └── q_learning/                   # [已实现] 轻量 Q-learning 模型 JSON
        ├── .gitkeep                  # [已实现] 保留目录
        ├── player/                   # [已实现] 玩家 Q-learning 模型
        │   └── .gitkeep              # [已实现] 保留目录；实际 .json 被 gitignore 忽略
        └── enemy/                    # [已实现] 敌人 Q-learning 模型
            └── .gitkeep              # [已实现] 保留目录；实际 .json 被 gitignore 忽略
```

---

## 已实现功能

### 游戏核心

- `[已实现]` 4x4 标准 2048 棋盘。
- `[已实现]` 上、下、左、右移动与合并。
- `[已实现]` 非法动作判断。
- `[已实现]` 游戏结束判断。
- `[已实现]` 分数、步数、最大块记录。
- `[已实现]` 棋盘状态快照 `GameState`。

### 玩家策略

- `[已实现]` `human`：命令行手动输入玩家。
- `[已实现]` `random`：随机合法动作玩家。
- `[已实现]` `heuristic`：启发式玩家，按空格数、可合并数量、最大块、角落、单调性等评分。
- `[已实现]` `q_ai`：AI 模型玩家，默认加载 `models/q_learning/player/latest/player_q_model.json` 的 Q 模型。
- `[已实现]` 纯标准库线性 Q 模型，支持保存/加载 JSON 权重。
- `[已实现]` Q-learning 训练入口：`python src/main.py train-player --episodes 300 --enemy greedy`。

### 敌人策略

- `[已实现]` `random`：标准随机敌人，默认 90% 出 2、10% 出 4。
- `[已实现]` `greedy`：贪心敌人，枚举所有空格和 2/4，选择当前评分最坏局面。
- `[已实现]` `q_enemy`：敌对 AI，默认加载 `models/q_learning/enemy/latest/enemy_q_model.json` 的 Q 模型选择出块位置和值。
- `[已实现]` 敌人 Q-learning 训练入口：`python src/main.py train-enemy --episodes 300 --player heuristic`。

### 运行方式

- `[已实现]` `python src/main.py`：默认启动 GUI。
- `[已实现]` `python src/main.py gui --enemy greedy`：显式启动 GUI。
- `[已实现]` `python src/main.py play --enemy random`：命令行手动玩。
- `[已实现]` `python src/main.py auto --player heuristic --enemy random --episodes 100`：命令行批量跑实验。
- `[已实现]` `python src/main.py auto --player q_ai --enemy random --episodes 100`：命令行评估 AI 模型玩家。
- `[已实现]` `python src/main.py train-player --episodes 300 --enemy random`：训练 AI 模型玩家。
- `[已实现]` `python src/main.py train-enemy --episodes 300 --player heuristic`：训练敌对 AI。

### GUI

- `[已实现]` 中文 Tkinter 主界面。
- `[已实现]` 左侧棋盘、右侧控制面板。
- `[已实现]` 右侧顶部可切换 `对局`、`模型训练`、`单项评估平台`、`训练评估平台` 四个工作区。
- `[已实现]` 棋盘 Canvas 绘制。
- `[已实现]` 方块滑动动画。
- `[已实现]` 方向键 / WASD 控制。
- `[已实现]` 重新开始。
- `[已实现]` AI 单步。
- `[已实现]` 自动播放。
- `[已实现]` 对局工作区内切换本局敌人类型。
- `[已实现]` GUI 内批量实验运行器。
- `[已实现]` GUI 内可调玩家、敌人、局数、随机种子、输出目录；单项评估 CSV 文件名固定为 `<player>_vs_<enemy>.csv`。
- `[已实现]` 单项评估可选择训练成果、自动玩家或自动敌人进行对比。
- `[已实现]` GUI 内可选择 `AI 模型玩家` 运行实验。
- `[已实现]` GUI 对局控制区可选择 `AI 模型玩家` 执行 AI 单步/自动播放。
- `[已实现]` GUI 内模型训练器，可设置训练对象、对手、训练局数、随机种子、参考模型、继续训练和输出目录。
- `[已实现]` GUI 内模型训练器可切换训练对象：`玩家 AI` / `敌对 AI`。
- `[已实现]` 训练 `玩家 AI` 时隐藏玩家选择，只显示对手敌人；训练 `敌对 AI` 时隐藏敌人选择，只显示固定玩家。
- `[已实现]` GUI 内模型训练器可停止训练，保存未完成模型、`info.json` 和训练日志。
- `[已实现]` GUI 内继续训练会生成新的时间戳目录，并在新目录保存成功后移除旧未完成目录。
- `[已实现]` GUI 内可训练敌对 AI，并保存为时间戳成果，同时发布到对应 `latest/` 默认目录。
- `[已实现]` GUI 训练完成后自动重新加载 AI 模型。
- `[已实现]` GUI 内实验进度条与结果保存提示。

### 实验与记录

- `[已实现]` 单局自动对战 `run_episode`。
- `[已实现]` 多局实验 `run_experiment`。
- `[已实现]` 正式实验 CSV 默认输出到 `outputs/experiments/<timestamp>/`。
- `[已实现]` CSV 记录字段：

```text
episode,max_tile,score,steps,player_type,enemy_type,seed
```

- `[已实现]` CLI 和 GUI 共用同一套实验逻辑。
- `[已实现]` 每次训练在模型时间戳目录写入 `log.jsonl`，记录模型路径、关键指标、best/checkpoint 路径和完成/未完成状态。
- `[已实现]` 每次评估在 CSV 同目录写入 `log.jsonl`。
- `[已实现]` 日志文件统一命名为 `log.jsonl`；同目录多日志时使用 `xxx_log.jsonl` 前缀形式。
- `[已实现]` 结果管理窗口可清理选中结果的本地日志，仅保留最近一条。
- `[已实现]` 主界面可清除系统日志和错误日志，每个日志文件保留最近十条。
- `[已实现]` CLI 和 GUI 后台任务捕获异常时写入 `logs/errors/log.jsonl`。

### 结果管理服务

- `[已实现]` `results/management.py` 统一扫描训练成果、latest 模型和评估 CSV，输出 GUI 可展示的 `ManagedResult`。
- `[已实现]` 删除训练或评估产物前会校验路径必须位于项目和受管目录内，避免误删源码或项目外文件。
- `[已实现]` 删除产物后会同步清理训练、评估和错误 JSONL 中引用该产物的日志行。
- `[已实现]` 支持将历史完整训练成果发布为对应类型的 `latest/` 默认模型，未完成训练不能发布。
- `[已实现]` 支持压缩选中产物的本地日志，以及压缩系统/错误日志。

### 测试

- `[已实现]` 使用标准库 `unittest`，不依赖 pytest。
- `[已实现]` 规则测试。
- `[已实现]` 玩家/敌人合法性测试。
- `[已实现]` 自动对局与 CSV 测试。
- `[已实现]` 无参数默认 GUI 入口解析测试。
- `[已实现]` DQN 稳定性配置与控制器测试。
- `[已实现]` 结果管理服务测试。

---

## 未实现 / 后续计划

### 更强敌人

- `[未实现]` `minimax_enemy.py`：搜索敌人，考虑玩家后续动作。
- `[未实现]` alpha-beta 剪枝。
- `[未实现]` 可调搜索深度。
- `[未实现]` 敌人只选位置、数值仍按 2/4 概率生成的约束模式。

### 强化学习玩家

- `[已实现]` `players/q_player.py`：轻量 Q-learning 模型玩家。
- `[已实现]` `models/q_learning/player.py`：玩家线性 Q 模型。
- `[已实现]` `train/q_learning/player.py`：Q-learning 玩家训练循环。
- `[已实现]` `players/dqn_player.py`：基于 PyTorch 的深度 DQN 玩家。
- `[已实现]` `models/dqn_network.py`：PyTorch MLP DQN 网络。
- `[已实现]` `train/dqn/checkpoints.py`：DQN checkpoint 保存、加载和 state_dict 共享工具。
- `[已实现]` `train/dqn/replay_buffer.py`：DQN 经验回放池。
- `[已实现]` `train/dqn/player.py`：深度 DQN 玩家训练循环。
- `[已实现]` `train/dqn/stability.py`：DQN 稳定性控制，支持 best checkpoint、滚动 checkpoint、回滚与学习率调整。
- `[已实现]` 自动选择 `cuda` / `cpu` 训练设备。
- `[已实现]` 默认依赖切换为 GPU 版 PyTorch CUDA 12.8，CPU 版保留为 `requirements-cpu.txt`。
- `[未实现]` `players/ppo_player.py`。
- `[未实现]` `models/ppo_network.py`。
- `[已实现]` 棋盘 `log2(tile)` 特征编码，当前用于线性 Q 模型。

### 强化学习敌人

- `[已实现]` `enemies/q_enemy.py`：轻量 Q-learning 敌对模型。
- `[已实现]` `models/q_learning/enemy.py`：敌人线性 Q 模型。
- `[已实现]` `train/q_learning/enemy.py`：固定玩家后训练敌人。
- `[已实现]` `enemies/dqn_enemy.py`：PyTorch 深度 DQN 敌人。
- `[已实现]` `train/dqn/enemy.py`：深度 DQN 敌人训练循环。
- `[已实现]` 敌人 DQN 输出 32 个出块动作，并对非法出块做动作屏蔽。
- `[已实现]` 敌人动作空间：16 个格子 x 2 种数值。
- `[已实现]` 合法出块动作过滤，相当于基础 action mask。
- `[未实现]` PPO 敌人训练。

### 自博弈与评估

- `[未实现]` 玩家 AI 与敌人 AI 轮流训练。
- `[未实现]` 模型版本管理：`player_vN`、`enemy_vN`。
- `[未实现]` 锦标赛评估矩阵。
- `[未实现]` 多玩家/多敌人横向对比。
- `[未实现]` 达到 128/256/512/1024/2048 的概率统计。

### 可视化与输出

- `[未实现]` 训练曲线图。
- `[未实现]` 最大块分布图。
- `[未实现]` 对战热力图。
- `[未实现]` 实验报告生成。
- `[未实现]` GUI 内查看历史 CSV。
- `[未实现]` 对局回放。

### 配置化

- `[已实现]` `configs/paths.yaml`：共享模型路径、正式输出目录、系统日志与错误日志路径。
- `[已实现]` `configs/evaluate/default.yaml`：批量评估默认参数。
- `[已实现]` `configs/ui/default.yaml`：GUI 初始默认值。
- `[已实现]` `configs/train/player_q.yaml`：Q-learning 玩家训练默认参数。
- `[已实现]` `configs/train/enemy_q.yaml`：Q-learning 敌人训练默认参数。
- `[已实现]` `configs/train/player_dqn.yaml`：DQN 玩家训练默认参数。
- `[已实现]` `configs/train/enemy_dqn.yaml`：DQN 敌人训练默认参数。
- `[已实现]` `src/config/__init__.py`：配置加载器；可使用 PyYAML，未安装时使用内置简易 YAML 解析。
- `[未实现]` `configs/self_play.yaml`。

---

## 后续开发建议

建议下一步按这个顺序推进：

1. 优化 `greedy` 敌人的出 2/4 策略，避免永远偏向 4。
2. 增加 `minimax_enemy.py`，先做 depth=1 或 depth=2。
3. 增加评估指标汇总，例如平均最大块、平均分、达到指定 tile 的比例。
4. 增加 GUI 内 CSV 结果摘要。
5. 继续调参深度 DQN：更大的网络、更长训练、训练曲线和模型对比。
