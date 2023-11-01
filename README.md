from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Мысалы, синтетикалық деректер жинағын жасаңыз
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Деректерді оқу және сынақ жинақтарына бөлеміз
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Негізгі классификаторды құру (мысалы, шешім ағашы)
from sklearn.tree import DecisionTreeClassifier
base_classifier = DecisionTreeClassifier(max_depth=1)

# AdaBoost классификаторын жасау
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)

# AdaBoost үлгісін жаттығу деректеріне үйрету
adaboost_classifier.fit(X_train, y_train)

# Сынақ деректері арқылы сыныптарды болжау
y_pred = adaboost_classifier.predict(X_test)

# Модельдің дәлдігін бағалау
accuracy = accuracy_score(y_test, y_pred)
print(f'Дәлдік: {accuracy}')

