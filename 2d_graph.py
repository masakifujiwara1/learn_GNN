import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt

# グラフ特徴の定義
# ここでは一括で乱数を用いて定義します．
# あとあとの図示で差がわかりやすいように+1しています
num_node = 5
in_channels = 4
X = np.random.randn(num_node, in_channels) + 1

# エッジの定義
E = [[0, 1], [0, 2], [0, 4], [1, 2], [2, 3]]

# 向きを考慮しないので逆向きのエッジを定義
reversed_E = [[j, i] for [i, j] in E]

# エッジを足します
new_E = E + reversed_E

# エッジから隣接行列を作成する関数を定義します
def edge2mat(E, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in E:
        A[j, i] = 1
    return A

# 自己ループ
I = [[i, i] for i in range(num_node)] 

# 隣接行列の作成
A = edge2mat(new_E + I, num_node)  # ノードの数は5

new_X = A @ X

# 結果の出力
print(f"エッジ: {new_E}")
print(f"入力サイズ: {X.shape}")
print(f"隣接行列サイズ: {A.shape}")

d_ii = np.sum(A, 0)
print(f"隣接行列サイズ: {d_ii}")

# fig, ax = plt.subplots(1, 2, width_ratios=[4, 5])
fig, ax = plt.subplots(1, 2)
ax[0].pcolor(X, cmap=plt.cm.Blues)
ax[0].set_aspect('equal', 'box')
ax[0].set_title('X', fontsize=10)
ax[0].invert_yaxis()

ax[1].pcolor(new_X, cmap=plt.cm.Blues)
ax[1].set_aspect('equal', 'box')
ax[1].set_title('new_X', fontsize=10)
ax[1].invert_yaxis()
plt.show()