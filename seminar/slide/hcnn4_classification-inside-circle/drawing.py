import matplotlib.pyplot as plt
import csv

# CSVファイルからデータを読み込んで2次元配列に格納する関数
def read_csv_to_2d_array(file_name):
    data = []
    with open(file_name, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            # 各行のデータをリストとして追加
            data.append([float(val) for val in row])
    return data

# ファイル名
file_name = 'circle_.csv'

# CSVファイルからデータを2次元配列に読み込む
data_array = read_csv_to_2d_array(file_name)

data = data_array

# x座標、y座標、ラベルを別々のリストに分割
x_values = [point[0] for point in data]
y_values = [point[1] for point in data]
labels = [point[2] for point in data]

# ラベルに応じて色を設定
colors = ['blue' if label == 0.0 else 'orange' for label in labels]

# 散布図をプロット
plt.scatter(x_values, y_values, c=colors, s=1)

# プロットの設定
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D plane')
plt.axis('equal')

# プロットを表示
plt.show()
