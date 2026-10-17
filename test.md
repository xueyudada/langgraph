**填写要求**

一、学位论文的中期检查是加强硕士研究生学位论文工作过程管理的重要环节，是对研究生学位论文工作的阶段性检查。

二、中期检查表由硕士研究生本人如实填写，导师对所填写的内容进行审核和检查，并给出结果评价。

三、本表填好后，纸质版提交学院研究生教学秘书，由学院组织专家（ **专业学位研究生中期检查应有行业专家参与** ）进行检查。

四、检查通过后，尽快在研究生信息管理系统内，申请中期考核，并上传签字、盖章的扫描文档，纸质版由研究生学籍所在学院统一保管。

一、学位论文工作进展 **（专硕需汇报专业实践情况）**

**（一）文献调研与选题阶段（2024年9月—2025年11月）**

**完成了对分层联邦学习（Hierarchical Federated Learning, HFL）、Stackelberg博弈、多智能体深度强化学习（Multi-Agent Deep Reinforcement Learning, MADRL）等领域的文献调研，阅读国内外相关文献。重点梳理了联邦学习激励机制的三大类型：契约型激励机制（基于委托-代理理论的自我揭示型契约设计）、拍卖型激励机制（基于多维竞价的反向拍卖框架）以及博弈型激励机制（基于Stackelberg博弈的多层级定价策略），深入分析各类机制在分层联邦学习场景下的适用性与局限性。**

**通过文献综述，明确了现有研究在三个方面的核心不足：（1）多层级协同缺失—现有激励机制大多将簇头简化为无决策能力的数据中继节点，仍采用“云-端”直连的双层博弈建模，忽略了簇头作为独立利益主体的自利性诉求与中间协调价值；（2）数据价值挖掘不足—传统参与者选择机制多聚焦设备硬件能力（如算力、能量、竞价成本），忽视数据多样性、样本类别完整性等数据质量维度的核心影响；（3）动态适配能力薄弱—传统方案在动态环境中普遍存在“策略僵化”问题，契约型机制受限于固定条款，拍卖型机制依赖静态数据质量假设，博弈型机制多采用静态梯度求解均衡点，无法实时跟进设备状态变化。**

**基于上述分析，确定了研究方向和论文选题——“基于多智能体强化学习的分层联邦学习激励机制研究”，并于2025年11月28日完成开题答辩。**

**（二）理论模型构建阶段（2025年12月—2026年4月）**

**1. 构建面向无人机辅助HFL的系统模型**

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000000_f4eba8d7bb0345860aa6ba0617abbc92d6618b52cd08889e94191a758034c50b.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000001_f4eba8d7bb0345860aa6ba0617abbc92d6618b52cd08889e94191a758034c50b.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000002_bd55c69908036d432cbf32a9b911940d2f74e97ac2d4cf6eb407fb17d5e14a32.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000003_bd55c69908036d432cbf32a9b911940d2f74e97ac2d4cf6eb407fb17d5e14a32.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000004_d1c28dd023de83b3b094513ad4512d3daeb54d07d35a199fff61c93621f17676.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000005_d1c28dd023de83b3b094513ad4512d3daeb54d07d35a199fff61c93621f17676.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000006_6cfa5b34f81348127e8f338e0190db3b64d73adff0d9b763ef2d2ad972bbc372.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000007_6cfa5b34f81348127e8f338e0190db3b64d73adff0d9b763ef2d2ad972bbc372.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000008_c11e151644997192fb277cfbf527431b0f6eed02e0da773be16101840ed8dc9f.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000009_c11e151644997192fb277cfbf527431b0f6eed02e0da773be16101840ed8dc9f.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000010_44f3ee0a0a52dc257b2a1325ca54159d9fb71868c865f5c2ab984751d00efce8.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000011_44f3ee0a0a52dc257b2a1325ca54159d9fb71868c865f5c2ab984751d00efce8.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000012_7175395d515784b9f1609896e40d68670ac7e190e932721fdc20b3ee58e8aa2a.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000013_7175395d515784b9f1609896e40d68670ac7e190e932721fdc20b3ee58e8aa2a.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000014_1c412e4659a95c72ecc0875c8e408517f34e39a01315d3cca2cd73f09561ea15.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000015_1c412e4659a95c72ecc0875c8e408517f34e39a01315d3cca2cd73f09561ea15.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000016_912c79ecdaf7b6ada93951945cea96f3c62a5a4a5cbdb0d0f9e260465852349a.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000017_912c79ecdaf7b6ada93951945cea96f3c62a5a4a5cbdb0d0f9e260465852349a.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000018_750304c22be22feb089036343a9ba5ed33d770dec542e33fab805d0878604f3c.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000019_750304c22be22feb089036343a9ba5ed33d770dec542e33fab805d0878604f3c.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000020_61e3f7fecddfdaf2b3e756bfd051810146be2e39e0053e9caabc72c2ba4b67bf.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000021_61e3f7fecddfdaf2b3e756bfd051810146be2e39e0053e9caabc72c2ba4b67bf.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000022_8493208ff8f51ca1eab021a588dd3ca03fa431d66d2e501fddaf7f72dca4e320.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000023_8493208ff8f51ca1eab021a588dd3ca03fa431d66d2e501fddaf7f72dca4e320.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000024_189987f59d1582524417a53c2364006874f4cd5fee1d37709ef1e886d716245a.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000025_189987f59d1582524417a53c2364006874f4cd5fee1d37709ef1e886d716245a.png)

**针对基础设施匮乏环境下的协同训练需求，构建了包含边缘服务器（ES）、K个工作单元（WUs）和N架能量受限无人机（UAVs）的三层系统架构。在工作单元层，精确建模了每个设备k的本地计算能耗（，与CPU频率、本地训练轮次、数据量相关）和通信能耗（，与模型参数大小、发射功率、信道增益相关）。在簇头层，建模了无人机的悬停能耗（）和任务相关能耗（包括聚合计算和回传通信能耗）。在云中心层，定义了全局模型的参数初始化方式和资源分配机制（总资源和激励预算的分配）。**

**2. 构建面向HFL架构的三层Stackelberg博弈模型**

**将云中心（CC）定义为顶层领导者（Leader），无人机簇头（CHs）定义为一级跟随者（Follower I），工作单元（WUs）定义为二级跟随者（Follower II）。精确定义了各层级的策略空间与效用函数：**

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000026_8cedcc744aed9397345d9e50f09a4f9724a681a9c814bf855a2e37df9bce93f8.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000027_8cedcc744aed9397345d9e50f09a4f9724a681a9c814bf855a2e37df9bce93f8.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000028_628fdaa587c08aff789d85f33e8b9590d42e21e063108d49dfa89e428a8b0d63.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000029_628fdaa587c08aff789d85f33e8b9590d42e21e063108d49dfa89e428a8b0d63.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000030_247e3258263e4d3cd59757c465d8060e08bc29fa7e51e60411683d9fbfc1ed0a.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000031_247e3258263e4d3cd59757c465d8060e08bc29fa7e51e60411683d9fbfc1ed0a.png)

- **CC的策略空间涵盖三个维度：激励分配、资源分配和参与者选择权重，受总预算和总资源约束；CC的效用函数综合考虑了质量加权性能增益、模型精度提升、资源分配成本、激励分配成本以及跨簇模型散度惩罚五个因素。**
![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000032_1778800cb09d16482c62be74c2c6ed2ce75a8bbaaeed117c878e3f51e022183f.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000033_1778800cb09d16482c62be74c2c6ed2ce75a8bbaaeed117c878e3f51e022183f.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000034_43ec62232cd3ac7e4c7bb575787f79ca3baddcd5e1b3de9d175c235a24db22e9.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000035_43ec62232cd3ac7e4c7bb575787f79ca3baddcd5e1b3de9d175c235a24db22e9.png)
- **CHs的策略空间包括下游竞价价格和跨簇协作请求；效用函数平衡了CC激励收益、资源收益、跨簇协作增益与竞价成本、能量消耗之间的关系。**
![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000036_140aa2a4a44bf52de874ef08c0d1151ad6444566ba39d66d6c025b0c7aa8f9fa.png)

![Image](F:\Learning\LangGraph\myLanggraph\test_artifacts\image_000037_140aa2a4a44bf52de874ef08c0d1151ad6444566ba39d66d6c025b0c7aa8f9fa.png)
- **WUs的唯一策略为参与概率；效用函数定义为激励收入减去能量成本和机会成本。**