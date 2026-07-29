from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veriyi yükle
iris = load_iris()

# Eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Modeli eğit
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test et ve doğruluğu yazdır
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
