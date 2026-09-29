# 本轮联调证据

全部为CPU合成数据；三轮提案由助手逐次读取真实dev结果后编写，非Naive/ARIS实测。train/dev/test图形高度重合，仅验证工程。

- campaign/：完整决策状态、冻结后最终测试。
- runs/：4次真实ISP训练及独立dev重载的命令、配置、日志与分数。
- proposal_*.json / next_*.json：3次先看结果、再改recipe的记录。
- selected_checkpoint/：冻结候选的实际权重与配套配置。
- pytest.log / repeatability.json：测试和重复评测数值误差。

记录含当时绝对路径；本机复跑请重新生成fixture并配置路径。原始模型/数据资源不包含在本目录；合成fixture可由overlay/scripts/make_smoke_data.py生成。
