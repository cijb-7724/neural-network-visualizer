from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
# Irisデータの読み込み
iris = datasets.load_iris()
X = iris.data
y = iris.target
# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify=y_train)

# モデルの構築・学習が2行 ( ﾉД`)ｼｸｼｸ… ｽﾞﾙｲﾖ
model = MLPClassifier(hidden_layer_sizes=(10,10), activation='relu', solver='adam', max_iter=500)
model.fit(X_train, y_train)

# 評価
train_accuracy = model.score(X_train, y_train)
val_accuracy = model.score(X_val, y_val)
test_accuracy = model.score(X_test, y_test)

print(f"Training accuracy: {train_accuracy}")
print(f"Validation accuracy: {val_accuracy}")
print(f"Test accuracy: {test_accuracy}")