# CI & Tests related

## Work Flow:

Commit -> Auto Test (UT, Integration Test, E2E Test)-> Build -> Second Round Test -> 

## Main method - Github actions

### Concepts:
Workflow:
Job:
Step:
Action

### Requirements

#### Runner

Need a server to run test: 

- Github provides, or
- Our own server

Pricing for Github standard runners:

Standard runners

|Operating system	|Per-minute rate (USD)|
|--|--|
|Linux 1-core|	$0.002|
|Linux 2-core|	$0.008|
|Windows 2-core	|$0.016|
|macOS 3-core or 4-core (M1 or Intel)|	$0.08|


### Action files

Each file is end with `.yaml` under `.github/workflows`.



[Github Official Action Marketplace](https://github.com/marketplace?type=actions)

[Awesome Actions](https://github.com/sdras/awesome-actions)












References:

- [Github Runners](https://docs.github.com/zh/actions/concepts/runners)
- [Github Actions](https://docs.github.com/zh/actions)
- [vLLM](https://docs.vllm.com.cn/en/latest/contributing/model/tests.html)
- [Juejin Link](https://juejin.cn/post/7388278660148609074)
- [Zhihu Link](https://zhuanlan.zhihu.com/p/250534172)
- [Link](https://blog.csdn.net/qq_36697163/article/details/140383476)