import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def visualize_distribution(df, columns):
    """Visualizar distribución de frecuencias de variables numéricas"""
    for column in columns:
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.histplot(df[column], bins=20, kde=True)
        plt.title(f'Distribución de {column}')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        plt.savefig(f'Distribución de {column}.png')
        plt.show()

def select_numeric_columns(df):
    """Seleccionar columnas numéricas"""
    return df.select_dtypes(include='number').columns

def load_data(file_path):
    """Cargar los datos desde un archivo CSV"""
    return pd.read_csv(file_path)

def train_linear_regression_model(X, y, test_size=0.3, random_state=42):
    """Entrenar un modelo de regresión lineal"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg, X_test, y_test

def evaluate_regression_model(regressor, X_test, y_test):
    """Evaluar un modelo de regresión"""
    y_pred = regressor.predict(X_test)
    r_squared = regressor.score(X_test, y_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return y_pred, r_squared, rmse

def plot_predictions_vs_actual(y_test, y_pred, title):
    """Graficar las predicciones vs los valores reales"""
     
    sns.set_style("whitegrid")
    sns.scatterplot(x=y_test, y=y_pred)
    for i in range(len(y_test)):
        plt.plot([y_test[i], y_test[i]], [y_test[i], y_pred[i]], '-', color='blue', alpha=0.2)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.title(title)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.savefig(f'Gráfico de residuos {title}.png')
    plt.show()
   

def main():
    # Cargar los datos
    sales_df = load_data('advertising_and_sales_clean.csv')

    # Seleccionar columnas numéricas
    numeric_cols = select_numeric_columns(sales_df)
    print("Columnas numéricas:", numeric_cols)

    # Visualizar la distribución de variables numéricas
    visualize_distribution(sales_df, numeric_cols)
    
    # Calcular la suma de cada columna numérica
    sum_df = sales_df[['tv', 'radio', 'social_media', 'sales']].agg('sum')

    # Crear un DataFrame con los datos de suma
    sum_data = {'Variable': sum_df.index, 'Suma': sum_df.values}
    sum_df = pd.DataFrame(sum_data)

    # Mostrar el DataFrame con las sumas por variable
    print(sum_df)

    # Configuración de estilo de seaborn
    sns.set_style("whitegrid")

    # Crear el gráfico de barras utilizando Seaborn
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(data=sum_df, x='Variable', y='Suma', palette='viridis')
    plt.title('Suma de Valores por Variable')
    plt.xlabel('Variable')
    plt.ylabel('Suma')
    plt.xticks(rotation=45, ha='right')

    # Agregar etiquetas en las barras
    for index, row in sum_df.iterrows():
        barplot.text(index, row['Suma'], round(row['Suma'], 2), color='black', ha="center")

    plt.tight_layout()
    plt.show()

    # Divide los datos en características (X) y variable objetivo (y)
    X = sales_df.drop(["sales", "influencer"], axis=1).values
    y = sales_df["sales"].values

    # Entrenar un modelo de regresión lineal01
    
    reg, X_test, y_test = train_linear_regression_model(X, y)

    # Evalua el modelo
    y_pred, r_squared, rmse = evaluate_regression_model(reg, X_test, y_test)
    print("R^2:", r_squared)
    print("RMSE:", rmse)

    # Grafica las predicciones vs los valores r
    plot_predictions_vs_actual(y_test, y_pred, "Predicciones vs Valores Reales: Modelo Optimo")
    
    # Divide los datos para un peor ajuste,es decir sólo emplea las RSS
    XPoor = sales_df.drop(["sales", "influencer", "tv", "radio"], axis=1).values
     # Entrenar un modelo de regresión lineal    XPoor
    regPoor, X_testPoor, y_testPoor = train_linear_regression_model(XPoor, y)
    # Evalua el modelo
    y_predPoor, r_squaredPoor, rmsePoor = evaluate_regression_model(regPoor, X_testPoor, y_testPoor)
    print("R^2Poor:", r_squaredPoor)
    print("RMSEPoor:", rmsePoor)

    # Grafica las predicciones vs los valores r
    plot_predictions_vs_actual(y_testPoor, y_predPoor, "Modelo Pobre (Poor) ")
    
    
if __name__ == "__main__":
    main()

