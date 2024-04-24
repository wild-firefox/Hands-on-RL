import gym
env = gym.make("FrozenLake-v1", render_mode="human")
env = env.unwrapped  # 解封装才能访问状态转移矩阵P
env.reset()
env.render()  # 环境渲染,通常是弹窗显示或打印出可视化的环境

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print("冰洞的索引:", holes)
print("目标的索引:", ends)

for a in env.P[14]:  # 查看目标左边一格的状态转移信息
    print(env.P[14][a])

# SFFF
# FHFH
# FFFH
# HFFG
# 冰洞的索引: {11, 12, 5, 7}
# 目标的索引: {15}
# [(0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False),
#  (0.3333333333333333, 14, 0.0, False)]
# [(0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False),
#  (0.3333333333333333, 15, 1.0, True)]
# [(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True),
#  (0.3333333333333333, 10, 0.0, False)]
# [(0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False),
#  (0.3333333333333333, 13, 0.0, False)]