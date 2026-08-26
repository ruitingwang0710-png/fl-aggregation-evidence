# 结果冻结 · 2026-08-25

**本文件的作用**：论文里每一个数字，都必须能指回这份表。写作时不得凭记忆引用。

---

## 1. 环境

| | |
|---|---|
| OS | macOS |
| Python | 3.13.6 |
| flwr | 1.33.0 |
| numpy | 2.5.2 |
| 核心复现命令 | `python evidence.py` → `python align_check.py` → `python bit_exact_check.py` → `python check_ab.py` |

> 交叉验证：同一套装置在 numpy 2.4.x 与 2.5.2 上给出相同结果。这是**跨版本可复现证据**，不构成"所有环境一致"的结论。

---

## 2. 装置

5 个等权客户端（各 `num-examples = 10`），声明 Krum，`f = 1`，`m = 1`（选 1 个），单次受控聚合调用，无数据集，无随机性。
邻域大小 `k = n − f − 2 = 2`。

**两组 fixture：**

| | 客户端提交 |
|---|---|
| **coincident** | `[1,1,1,1]` `[1.1,1,1,1]` `[0.9,1,1,1]` `[1,1.1,1,1]` `[1,0.9,1,1]` |
| **divergent** | `[1,1,1,1]` `[1.1,1,1,1]` `[0.9,1,1,1]` `[1,1.1,1,1]` `[9,9,9,9]` |

coincident 的构造：**一个质心 + 两对对称 ±0.1 偏移**，使质心恰好等于全体均值，且质心同时是 Krum 分数最低者。

**Krum 分数（coincident，保留两位小数）**：`[0.02, 0.03, 0.03, 0.03, 0.03]` → 选中客户端 0。

**四个受控聚合世界**（全部声明 Krum(f=1)）：

| world | 声明 | 实际执行 | fixture |
|---|---|---|---|
| A | krum | krum | coincident |
| B | krum | **fedavg** | coincident |
| C | krum | **fedavg** | divergent |
| D | krum | krum | divergent |

---

## 3. 三个核心数字

| 量 | 值 |
|---|---|
| **coincident 上 Krum 与 FedAvg 的输出差** | **`0.0`；uint64 视图及完整数组字节均相同** |
| divergent 上 Krum 与 FedAvg 的输出差 | `1.62` |
| 独立 NumPy 实现 vs Flower 实现（两条规则 × 两组 fixture，共 4 组对比） | 全部 `0.0` |

第三行是 C2「独立重放」成立的前提：**若独立实现与被测实现不一致，重放就不是在重放同一条规则。**

---

## 4. 判定矩阵（`results/summary.csv`，51 行）

| case | C1 声明 | C2 重放 | C3 执行路径 |
|---|---|---|---|
| A/E0 | consistent | unable_to_determine | unable_to_determine |
| A/E1 | consistent | unable_to_determine | unable_to_determine |
| A/E2 | consistent | consistent | unable_to_determine |
| A/E3 | consistent | consistent | consistent |
| **B/E0** | consistent | unable_to_determine | unable_to_determine |
| **B/E1** | consistent | unable_to_determine | unable_to_determine |
| **B/E2** | consistent | **consistent** | unable_to_determine |
| **B/E3** | consistent | **consistent** | **contradicted** |
| C/E0 | consistent | unable_to_determine | unable_to_determine |
| C/E1 | consistent | unable_to_determine | unable_to_determine |
| C/E2 | consistent | contradicted | unable_to_determine |
| C/E3 | consistent | contradicted | contradicted |
| D/E0 | consistent | unable_to_determine | unable_to_determine |
| D/E1 | consistent | unable_to_determine | unable_to_determine |
| D/E2 | consistent | consistent | unable_to_determine |
| D/E3 | consistent | consistent | consistent |
| tampered/E3 | invalid_bundle | invalid_bundle | invalid_bundle |

**C2 的最大差值：** A `0.0` · **B `0.0`** · C `1.62` · D `0.0`

---

## 5. A 与 B 的证据包比较

| 证据级别 | A 与 B 是否逐字节相同 |
|---|---|
| E0（仅声明） | **完全相同** |
| E1（+ 输出） | **完全相同** |
| E2（+ 客户端提交） | **完全相同** |
| E3（+ 运行期记录） | 不同（差异文件：`execution.json`、`manifest.json`） |

---

## 6. 可写进论文的四句话（每句都指回上表）

1. **反例**：存在一组客户端提交，使 Krum(f=1) 与 FedAvg 的输出**位精确相同**（uint64 视图和完整数组字节相同，`max_abs_difference = 0.0`）。
2. **失效**：在该输入上，独立重放声明规则并与记录的输出比对，对一次**声明 Krum、实际调用 FedAvg** 的受控聚合返回 `consistent`，差值为精确的零——**精确相等或任何非负容差都无法区分。**
3. **不可区分**：世界 A（防御执行过）与世界 B（未执行）的证据包，在 E0、E1、E2 三级上**逐字节完全相同**；仅在加入运行期执行记录后才分开。
4. **对照**：同一检查程序在 divergent 输入上正确返回 `contradicted`（差 `1.62`）。**该协议并非全无效力，而是在特定输入上失效——而输入可以被选择。**

---

## 7. 边界（必须写进 Limitations）

1. **规模**：5 个合成客户端、单次受控聚合调用、单一规则对。不构成对该情形在真实部署中发生频率的估计。
2. **执行范围**：A–D 直接向 Flower 原版 strategy 方法传入 Flower Message 对象，不是四次端到端分布式训练；完整 `flwr run` 只用于留存观察。
3. **位精确的成因**：`1.1 + 0.9` 在 float64 下舍入回精确的 `2.0`，使两对 ±0.1 完全抵消。**这是关于这一组输入的性质，不保证换一组数仍成立。**
4. **C3 依赖 recorder 可信**：运行期调用栈由执行聚合的同一进程写入。它确立执行，**只在 recorder 本身可靠且未被绕过的前提下**。
5. **完整性不等于真实性**：`invalid_bundle` 只表示当前文件字节与所附 manifest 中的摘要不一致。manifest 本身没有认证且可被一起重写，因此该 verdict 只检出本实验的受控破坏，不证明修改时间、真实性或抗恶意篡改。
6. **留存行为的检索范围**：见第 8 节。结论基于关键词检索、调用点分析和一次默认配置下的本机运行，不构成“该版本不存在任何扩展或替代记录路径”的穷尽证明。

---

## 8. Flower 的消息留存行为（2026-08-25 查证，源码级）

原「待查」项已查清，且结论强于原假设。

| 位置 | 代码 | 是否受配置控制 |
|---|---|---|
| `superlink/grid/inmemory_grid.py:140`（模拟路径） | `self.state.delete_messages(message_ins_ids=message_ins_ids_to_delete)` | **否**。`pull_messages()` 中取走回复的同时即删除指令消息 |
| `server/superlink/utils.py:38` | 运行状态为 `FINISHED` 时调用 `state.cleanup_run(run_id)` | **否** |
| `server/superlink/linkstate/linkstate.py:140` | `cleanup_run()` = `delete_messages(...)` + `object_store.delete_objects_in_run(...)` + `delete_sessions_in_run(...)` | **否** |

`delete_messages` 定义在抽象基类 `LinkState` 上，SQLite 与内存两个后端均实现；其 SQL 同时删除 `message_ins` 与 `message_res`
（`sql_linkstate.py:616, 622`）。

检索 `retention` / `retain` / `keep_message` / `persist_message` 未发现可关闭该行为的配置项。

**可写进论文的表述：**

> 在本实验使用的 flwr 1.33.0 本地模拟路径中，`pull_messages()` 将回复返回给 ServerApp 后，会在同一次调用中删除 LinkState 中对应的指令／回复记录；run 结束时再清理剩余的 run-scoped 消息和对象。这些删除点不读取留存配置。因此，只依赖默认 SuperLink 持久化状态的事后审阅者拿不到重放所需的客户端提交；若要重放，必须在数据仍位于内存时另行记录。

---

## 9. 复现方式

```bash
cd ~/fl-walkthrough
python evidence.py      # 生成 evidence/ 与 results/summary.csv
python align_check.py   # 独立 NumPy 实现与 Flower 对齐
python bit_exact_check.py  # 断言 uint64 视图与完整数组字节相同
python check_ab.py      # 打印 C2 差值并直接比较 A/B 的完整文件字节

.venv/bin/flwr run coincident-app local-simulation --stream
python inspect_state.py # 检查完成后的默认 SuperLink 状态
```

`evidence/*/truth.json` 存放各世界的真值，**位于 bundle 目录之外**，验证器不读取。
