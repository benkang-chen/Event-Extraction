# kg-2019-baseline
2019年百度的三元组抽取比赛（ http://lic2019.ccf.org.cn/kg ），一个baseline


## 模型
用BiLSTM做联合标注，先预测subject，然后根据suject同时预测object和predicate，标注结构是“半指针-半标注”结构，以前也曾介绍过（ https://kexue.fm/archives/5409 ）

