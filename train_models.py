import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, mean_squared_error, mean_absolute_error

def main():
    # Detect absolute path of the script directory to find files reliably
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'titanic.csv')
    
    if not os.path.exists(csv_path):
        # Fallback to current working directory
        csv_path = 'titanic.csv'
        
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} no encontrado.")
        return
        
    print(f"Cargando dataset desde: {os.path.abspath(csv_path)}")
    df = pd.read_csv(csv_path)
    print("Dimensiones del dataset:", df.shape)

    # 1. Preparación de Datos (Fase 2 de CRISP-ML)
    # Imputación de edad con la mediana
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)

    # Imputación de Embarked con la moda
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)

    # Imputación de Tarifas con la mediana
    fare_median = df['Fare'].median()
    df['Fare'] = df['Fare'].fillna(fare_median)

    # Codificación de variables categóricas
    df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked_encoded'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # 2. Entrenamiento del Clasificador (Regresión Logística para Supervivencia)
    features_clf = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
    X_clf = df[features_clf]
    y_clf = df['Survived']

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_clf, y_train_clf)

    y_pred_clf = clf.predict(X_test_clf)
    acc = accuracy_score(y_test_clf, y_pred_clf)
    cm = confusion_matrix(y_test_clf, y_pred_clf)

    print("\n==============================================")
    print("MÉTRICAS DE CLASIFICACIÓN (Logistic Regression)")
    print("==============================================")
    print(f"Exactitud (Accuracy): {acc:.4f}")
    print("Matriz de Confusión:")
    print(cm)

    # 3. Entrenamiento del Regresor (Regresión Lineal para Tarifas)
    # Predictor de Fare usando Pclass, Sex_encoded, Age, SibSp, Parch, Embarked_encoded, y Survived
    features_reg = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Embarked_encoded', 'Survived']
    X_reg = df[features_reg]
    y_reg = df['Fare']

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train_reg, y_train_reg)

    y_pred_reg = reg.predict(X_test_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)

    print("\n===========================================")
    print("MÉTRICAS DE REGRESIÓN (Linear Regression)")
    print("===========================================")
    print(f"Coeficiente de Determinación (R2 Score): {r2:.4f}")
    print(f"Error Absoluto Medio (MAE): {mae:.4f}")
    print(f"Error Cuadrático Medio (MSE): {mse:.4f}")

    # 4. Guardar los modelos entrenados y los metadatos (Fase 3 de CRISP-ML)
    classifier_file = os.path.join(script_dir, 'model_classifier.pkl')
    regressor_file = os.path.join(script_dir, 'model_regressor.pkl')
    metadata_file = os.path.join(script_dir, 'model_metadata.pkl')

    with open(classifier_file, 'wb') as f:
        pickle.dump(clf, f)

    with open(regressor_file, 'wb') as f:
        pickle.dump(reg, f)

    metadata = {
        'age_median': age_median,
        'fare_median': fare_median,
        'embarked_mode': embarked_mode
    }
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    print("\n===========================================")
    print("MODELOS Y METADATOS GUARDADOS CON ÉXITO")
    print("===========================================")
    print(f"-> {os.path.basename(classifier_file)}")
    print(f"-> {os.path.basename(regressor_file)}")
    print(f"-> {os.path.basename(metadata_file)}")

if __name__ == '__main__':
    main()
