import numpy as np

from constants import constants

MAX_CHANGE = constants.DP_MAX_CHANGE  # 1時刻移動するときに水平方向にどれぐらい移動を許すか(hardな制約)
PENALTY = constants.DP_PENALTY  # 1時刻移動するときに水平方向にどれぐらい移動を許すか(ソフトな制約）. 0で無効＝ペナルティ無．
DEFAULT_DP_START_FRAME = constants.DEFAULT_DP_START_FRAME
DEFAULT_DP_END_FRAME = constants.DEFAULT_DP_END_FRAME

# MAX_CHANGE should be an odd integer
assert MAX_CHANGE % 2 == 1


def create_dummy_matrix():
    mat = np.random.random(size=(8, 6)) / 1.2
    for row in mat:
        row[0] = 1.0
    for row, col in enumerate([2, 2, 3, 4, 3, 2, 3, 1]):
        mat[row, col] = 1.0
    print("Dummy Matrix")
    print(mat)
    return mat


def calc_dp(D, start=DEFAULT_DP_START_FRAME, end=DEFAULT_DP_END_FRAME):
    D = D[:, start:end]
    # Initialization
    T, K = D.shape
    G = np.zeros_like(D)  # 累積スコア
    B = np.empty_like(D, dtype=int)  # バックポインタ
    # Initial frame t=0
    G[0] = D[0]

    # DP recursion
    for t in range(T):
        for k in range(K):
            max_val = -1.0
            max_pos = None
            # 最大値探索
            for kk in range(max(k - MAX_CHANGE // 2, 0), min(k + MAX_CHANGE // 2, K)):
                if max_val < G[t - 1, kk]:
                    max_val = G[t - 1, kk] - PENALTY * abs(k - kk)
                    max_pos = kk
            G[t, k] = D[t, k] + max_val
            B[t, k] = max_pos

    # 最終時刻での最大値
    max_pos = np.argmax(G[-1])

    # バックトラック
    ret = []
    for t in reversed(range(T)):
        ret.append(max_pos)
        max_pos = B[t, max_pos]
    return np.array(ret)[::-1] + start


if __name__ == "__main__":
    D = create_dummy_matrix()
    print(calc_dp(D, 1, 6))
