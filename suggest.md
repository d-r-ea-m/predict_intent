建议走一条统一主线：车道语义软标签 + 意图查询解码 + 类型原型残差解码。这样三件事是耦合的，不是各自打补丁。

1. 你现在的意图标签怎么改进
你当前标签确实偏粗：在 build_style_intent_labels.py 里，本质是用最后有效未来点做一次横向投影阈值分类（见 build_style_intent_labels.py, build_style_intent_labels.py, build_style_intent_labels.py）。

建议改成同一套三分类但更语义化的打标：
- 用多时域而非单终点：1秒、3秒、全未来三段共同投票。
- 引入车道语义分支特征：当前车道切向变化、未来轨迹对应车道分支方向。
- 输出软分布再取硬标签，减少边界噪声。  
  $$s_i=w_1 f_{lat,i}+w_2 f_{\Delta\psi,i}+w_3 f_{lane,i},\quad p_i=\mathrm{softmax}(s_i/T),\quad y=\arg\max_i p_i$$
- 训练时用 $p_i$ 作为软监督或样本权重，标签仍保持左/直/右三类不变。

2. 你说的 MLP + Ii-1 还有提升空间吗
有，而且你当前实现其实比这个更简化：意图头是单线性层 qcnet.py，再 softmax 后送入解码器 qcnet.py, qcnet.py。解码器是向量相加融合 qcnet_decoder.py, qcnet_decoder.py，损失是并列加权 qcnet.py。

推荐升级为意图查询解码器（替代纯 MLP）：
- 3个意图查询 token，做自注意力 + 场景交叉注意力，再输出意图 logits 与意图嵌入。
- 用门控残差更新 $I^{l-1}\rightarrow I^l$，避免浅层 MLP 表达瓶颈。
- 加一个意图-轨迹一致性损失，让意图真的约束 mode：  
  $$L=L_{traj}+\lambda_1 L_{intent}+\lambda_2\,\mathrm{KL}\!\left(p(intent)\,\|\,\sum_{m\in\mathcal M_i}\pi_m\right)$$

3. kmeans 先验轨迹怎么嵌入更好，怎么和意图结合
你这部分数据形态很适合做原型残差解码：
- 先验中心是按类型聚类得到的 kmeans_motion.py，并保存为 npy kmeans_motion.py。
- 目前主模型没有读取先验文件的入口参数（见参数注册块 qcnet.py），所以先验尚未进入解码。
- 另外你当前 kmeans_motion 默认用 val 做聚类 kmeans_motion.py, kmeans_motion.py，如果用于训练先验会有评估泄漏风险；应改为仅 train 拟合、val/test只使用。
- 另外先验轨迹生成可以考虑与意图真实标签结合，如每个标签生成两条轨迹，以避免直行轨迹过多导致问题
- 另外目前数据中有大量静止目标，需要用阈值进行筛选，不让静止目标干扰先验轨迹聚类

最优嵌入方式：
- 类型检索：按 agent type 取对应 K 条原型。
- 原型编码：复用现有轨迹编码器 qcnet_decoder.py, qcnet_decoder.py 得到 prototype token。
- 残差预测：不是从零预测，而是  
  $$\hat{\tau}_m=c_m+\Delta\tau_m$$
  其中 $c_m$ 是原型中心，$\Delta\tau_m$ 是网络残差。
- 与意图结合：用分层门控  
  $$p(m)=\sum_i p(i)\,p(m|i)$$
  并用意图-模式一致性约束联合训练。

补充一点：你的 kmeans_plan.py 是 nuScenes 口径（命令导航），与当前 AV2 主流程不是同一数据协议，建议先聚焦 kmeans_motion.py 这条线。

下一步建议按这个顺序推进：
1. 先改标签定义与监督（否则后面网络改造收益会被噪声标签吞掉）。
2. 再上意图查询解码器与一致性损失。
3. 最后接入 kmeans 原型做残差解码并和意图做分层门控。