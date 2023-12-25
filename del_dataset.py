import pandas as pd

# ファイルを読み込んでデータフレームに変換する
file_path = "./dataset/obsmat.txt"  # ファイルのパスを指定してください
data = pd.read_csv(file_path, header=None)  # 列名がない場合はheader=Noneとします

print(data[:10])

# 削除する列のインデックスをリストで指定する
columns_to_delete = [3, 6]  # 削除する列のインデックスを指定してください

# 特定の複数列を削除する
data.drop(columns=columns_to_delete, inplace=True, axis=1)  # axis=1は列を指定していることを示します

# 新しいファイルに書き出す
output_file_path = "./del_dataset/output_file.txt"  # 書き出すファイルのパスを指定してください
data.to_csv(output_file_path, header=False, index=False)

print(f"{columns_to_delete} 列を削除して新しいファイルに書き出しました。")
