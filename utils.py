from collections import deque
import numpy as np
import sgfmill
from sgfmill import sgf, boards


class SGFParseRankPrediction:
    """
    解析 SGF 棋譜並提取包含「落子順序編號」、「氣數分級」與「眼位死活」的高階圍棋特徵。
    依據目標玩家執色 (color='B' 或 'W') 將視角統一對齊為「第一人稱視角 (我方 vs 對手)」：
      - 盤面數值  1: 我方（欲預測棋力的目標玩家）棋子
      - 盤面數值 -1: 對手棋子
    時序對齊方式：
      - 過濾 SGF 空盤根節點，嚴格從第 1 手開始取前 250 手，不足 250 手則於末端補零 (Post-padding)。
    輸出維度: (353, 19, 19)，dtype=np.uint8
      -   2 通道 (0 ~ 1)    : 全局第 1~250 手「我方 / 對手」落子順序編號平面 (數值 0 或 1~250)
      -  50 通道 (2 ~ 51)   : 25 個截面 (每 10 步) 的區間內「我方 / 對手」落子順序編號平面 (數值 0 或 1~10)
      -  50 通道 (52 ~ 101) : 25 個截面 (每 10 步) 的「我方 / 對手」全盤黑白狀態快照 (數值 0 或 1)
      - 250 通道 (102 ~ 351): 25 個截面 (每 10 步) 的「我方 / 對手」氣數與死活平面 (25 * 10，數值 0 或 1)
      -   1 通道 (352)      : 全 1 棋盤邊界常數平面 (AlphaGo Constant Ones Plane)
    """

    def __init__(self, history_length: int = 250, tactical_stride: int = 10):
        self.history_length = history_length
        self.tactical_stride = tactical_stride

    @staticmethod
    def build_perspective_swap_indices(history_length: int = 250, tactical_stride: int = 10) -> list:
        """
        產生將「黑/白固定視角」對調為「白方第一人稱視角 (我方=白, 對手=黑)」的 353 通道索引排列。
        """
        num_tactical = history_length // tactical_stride
        history_channels = 2 + num_tactical * 2 + num_tactical * 2  # 2 + 50 + 50 = 102
        total_channels = history_channels + num_tactical * 10 + 1   # 102 + 250 + 1 = 353
        idx = list(range(total_channels))

        # 1. 前 102 通道 (51 組 我方/對手 配對): 偶數與奇數互換
        for i in range(history_channels // 2):
            idx[i * 2] = i * 2 + 1
            idx[i * 2 + 1] = i * 2

        # 2. 戰術 250 通道 (25 個截面 * 10): [0~3] <-> [4~7], [8] <-> [9]
        base_offset = history_channels
        for s in range(num_tactical):
            b = base_offset + s * 10
            for k in range(4):
                idx[b + k] = b + 4 + k
                idx[b + 4 + k] = b + k
            idx[b + 8] = b + 9
            idx[b + 9] = b + 8

        # 3. 最後 1 通道 (352，全 1 邊界平面) 保持原位
        return idx

    def _prepare_board(self, board: sgfmill.boards.Board, target_color: str = 'b'):
        board_np = np.zeros((19, 19), dtype=np.int8)
        # 若目標玩家執黑 ('b')，黑子為 1 (我方)、白子為 -1 (對手)
        # 若目標玩家執白 ('w')，白子為 1 (我方)、黑子為 -1 (對手)
        self_sign = 1 if target_color == 'b' else -1
        # 直接遍歷盤面上的有子點，比雙迴圈掃描 361 點更快速
        for color, (row, col) in board.list_occupied_points():
            if color == 'b':
                board_np[col, row] = self_sign   # 與原版座標定義完全一致: board.get(x, y) -> board_np[y, x]
            elif color == 'w':
                board_np[col, row] = -self_sign
        return board_np

    def _compute_tactical_planes(self, board_np: np.ndarray) -> np.ndarray:
        """
        計算單一盤面的 10 個「氣數與死活」二元特徵平面:
        [0~3]: 我方連塊氣數 (1口氣, 2口氣, 3口氣, >=4口氣)
        [4~7]: 對手連塊氣數 (1口氣, 2口氣, 3口氣, >=4口氣)
        [8]  : 我方眼位/虎口控制點 (四方皆為我方棋子包圍之空點)
        [9]  : 對手眼位/虎口控制點 (四方皆為對手棋子包圍之空點)
        """
        planes = np.zeros((10, 19, 19), dtype=np.uint8)

        # 若為末端補零的空白棋盤，直接回傳全 0 矩陣以節省運算時間
        if not np.any(board_np):
            return planes

        visited = np.zeros((19, 19), dtype=bool)
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

        for y in range(19):
            for x in range(19):
                val = board_np[y, x]

                # 1. 若為空交叉點，檢查是否構成「基本眼位 / 禁手空間」
                if val == 0:
                    self_neighbors = 0
                    opp_neighbors = 0
                    total_neighbors = 0
                    for dy, dx in directions:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < 19 and 0 <= nx < 19:
                            total_neighbors += 1
                            n_val = board_np[ny, nx]
                            if n_val == 1:
                                self_neighbors += 1
                            elif n_val == -1:
                                opp_neighbors += 1
                    if self_neighbors == total_neighbors:
                        planes[8, y, x] = 1  # 我方完全包圍的眼位點
                    elif opp_neighbors == total_neighbors:
                        planes[9, y, x] = 1  # 對手完全包圍的眼位點
                    continue

                # 2. 若為棋子且尚未拜訪，使用 BFS 找出整個連塊 (Chain) 並計算共享氣數
                if not visited[y, x]:
                    visited[y, x] = True
                    queue = deque([(y, x)])
                    chain_stones = [(y, x)]
                    liberties = set()

                    while queue:
                        cy, cx = queue.popleft()
                        for dy, dx in directions:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < 19 and 0 <= nx < 19:
                                n_val = board_np[ny, nx]
                                if n_val == 0:
                                    liberties.add((ny, nx))  # 空點即為該連塊的「氣」
                                elif n_val == val and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    chain_stones.append((ny, nx))
                                    queue.append((ny, nx))

                    # 將氣數分為 1, 2, 3, >=4 四個戰術等級
                    num_lib = len(liberties)
                    lib_tier = min(num_lib, 4) - 1  # 對應索引 0, 1, 2, 3

                    if lib_tier >= 0:
                        channel_offset = 0 if val == 1 else 4
                        target_ch = channel_offset + lib_tier
                        for sy, sx in chain_stones:
                            planes[target_ch, sy, sx] = 1

        return planes

    def _extract_game_states(self, game, target_color: str = 'b'):
        board = sgfmill.boards.Board(19)
        board_states = []
        moves_info = []  # 記錄每一手: (is_self, (y, x) 座標或 None)

        for node in game.get_main_sequence():
            move = node.get_move()
            # 【修正問題 1】跳過 SGF 根節點 (move[0] is None)，確保索引 0 嚴格對應第 1 手落子
            if move[0] is None:
                continue

            color = move[0]
            is_self = (color == target_color)
            if move[1] is not None:
                row, column = move[1][0], move[1][1]
                board.play(row, column, color)
                coord = (column, row)  # 與 _prepare_board 的 board_np[col, row] 座標定義完全一致
            else:
                coord = None           # 虛手 (Pass)

            board_states.append(self._prepare_board(board, target_color))
            moves_info.append((is_self, coord))

            # 【修正問題 1】從第 1 手固定取前 history_length (250) 手，超過直接停止重播以節省時間
            if len(board_states) >= self.history_length:
                break

        return board_states, moves_info

    def _extract_features(self, board_states, moves_info):
        # 【修正問題 1】不足 250 手時改於「末端補零 (Post-padding)」，確保所有棋局時序通道對齊
        if len(board_states) < self.history_length:
            pad_count = self.history_length - len(board_states)
            padding = [np.zeros((19, 19), dtype=np.int8) for _ in range(pad_count)]
            history_states = board_states + padding
        else:
            history_states = board_states[:self.history_length]

        num_tactical = self.history_length // self.tactical_stride  # 25 個截面

        # 【修正問題 2】建立全局與每 10 步區間的「落子順序編號平面 (Move Number Encoding Planes)」
        # 1. 全局落子順序平面 (2, 19, 19): 通道 0 為我方、通道 1 為對手，數值為第幾手 (1~250)
        global_move_planes = np.zeros((2, 19, 19), dtype=np.uint8)
        # 2. 每 10 步區間落子順序平面 (50, 19, 19): 每截面 2 通道 (我方/對手)，數值為該 10 步內的第幾手 (1~10)
        local_move_planes = np.zeros((num_tactical * 2, 19, 19), dtype=np.uint8)

        for step_idx, (is_self, coord) in enumerate(moves_info[:self.history_length]):
            if coord is None:
                continue
            y, x = coord
            role_offset = 0 if is_self else 1

            # 寫入全局手數編號 (1 ~ 250)
            global_move_planes[role_offset, y, x] = step_idx + 1

            # 寫入所屬 10 步截面的區間手數編號 (1 ~ 10)
            interval_idx = step_idx // self.tactical_stride
            local_step_num = (step_idx % self.tactical_stride) + 1
            local_move_planes[interval_idx * 2 + role_offset, y, x] = local_step_num

        # 3. 每隔 tactical_stride (10) 步提取一次全盤黑白狀態快照 (共 25 * 2 = 50 通道)
        tactical_indices = range(self.tactical_stride - 1, self.history_length, self.tactical_stride)
        snapshot_planes = np.empty((num_tactical * 2, 19, 19), dtype=np.uint8)
        for s, idx in enumerate(tactical_indices):
            t = history_states[idx]
            snapshot_planes[s * 2] = (t == 1)      # 我方該截面全盤棋子
            snapshot_planes[s * 2 + 1] = (t == -1) # 對手該截面全盤棋子

        # 4. 每隔 tactical_stride (10) 步提取一次 10 通道的「氣數與死活」戰術截面 (共 25 * 10 = 250 通道)
        tactical_list = [self._compute_tactical_planes(history_states[idx]) for idx in tactical_indices]
        tactical_features = np.concatenate(tactical_list, axis=0)

        # 5. 1 個全 1 棋盤邊界常數平面 (1, 19, 19)
        border_plane = np.ones((1, 19, 19), dtype=np.uint8)

        # 串接所有特徵: 2 + 50 + 50 + 250 + 1 = 353 通道
        features = np.concatenate([
            global_move_planes,
            local_move_planes,
            snapshot_planes,
            tactical_features,
            border_plane
        ], axis=0)
        return features

    def features_from_sgf_content(self, game_content: str, color: str = 'B'):
        try:
            target_color = 'w' if str(color).strip().upper() == 'W' else 'b'
            game = sgfmill.sgf.Sgf_game.from_string(game_content)
            board_states, moves_info = self._extract_game_states(game, target_color=target_color)
            features = self._extract_features(board_states, moves_info)
            return features, None
        except Exception as e:
            return None, e
