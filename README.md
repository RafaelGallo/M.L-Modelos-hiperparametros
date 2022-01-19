# Machine-learning-Hiperparametros

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

![Logo](https://image.freepik.com/vetores-gratis/conceito-de-big-data-e-inteligencia-artificial-humanoide-de-inteligencia-artificial-com-rede-neural-inclui-tecnologia-de-big-data_1150-55388.jpg)


## Autores

- [@RafaelGallo](https://www.github.com/rafaelgallo)


## Objetivo do projeto de machine learning

Nesse reprositorio fiz alguns projetos de machine learning com hiperparâmetro.
hiperparâmetro são utilizado para não ter overfitting no modelo ele pode ser utlizado muito bem até mesmo pipeline.
Nesse projeto realizei um modelo de machine learning previsão ser a pessoa vai ter câncer. Outro modelo previsão de doenças cardíacas, diabetes.

# Projeto realizado

- Machine learning - Câncer hiperparâmetro
- Machine learning - Diabetes hiperparâmetro - Regressão logistica
- Machine learning - Doenças cardíacas

## Stack utilizada

**Programação** Python

**Leitura CSV**: Pandas

**Análise de dados**: Seaborn, Matplotlib

**Machine learning**: Scikit-learn





## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

Instalando a virtualenv

`pip install virtualenv`

Nova virtualenv

`virtualenv nome_virtualenv`

Ativando a virtualenv

`source nome_virtualenv/bin/activate` (Linux ou macOS)

`nome_virtualenv/Scripts/Activate` (Windows)

Retorno da env

`projeto_py source venv/bin/activate` 

Desativando a virtualenv

`(venv) deactivate` 

Instalando pacotes

`(venv) projeto_py pip install flask`

Instalando as bibliotecas

`pip freeze`


## Instalação

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install -c conda-forge pandas 
  conda install -c conda-forge scikitlearn
  conda install -c conda-forge numpy
  conda install -c conda-forge scipy

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```
    
## Demo modelo Hiperparâmetro

```bash
  # Carregando as bibliotecas 
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  # Carregando o dataset
  data = pd.read_csv("data.csv")
  
  # Visualizando os 5 primeiros itens
  data.head()

  # visualizando linhas e colunas com shape
  data.shape

  # Informações das variaveis
  data.info()

  # Análise de dados
  
  # Gráfico de barras
  plt.figure(figsize=(20, 10))
  plt.title("Total email real e spam")
  sns.countplot(base["Prediction"])
  plt.xlabel("SPAM REAL")
  plt.ylabel("Total")

  #Gráfico de pizza
  plt.figure(figsize=(20, 10))

  plt.pie(base.groupby("Prediction")['Prediction'].count(), labels=["SPAM", "REAL"], autopct = "%1.1f%%");
  plt.title("Total de email spam ou real")
  plt.legend(["SPAM", "REAL"])

  #Gráfico de scatterplot
  plt.figure(figsize=(15.8, 10))
  ax = sns.scatterplot(x="the", y="hou", data = base, hue ="Prediction")
  plt.title("Total email real e spam")
  plt.xlabel("SPAM REAL")
  plt.ylabel("Total")

  # Gráfico de histrograma
  plt.figure(figsize=(15.8, 10))
  ax = sns.histplot(x="the", y="hou", data = base, hue ="Prediction")
  plt.title("Total email real e spam")
  plt.xlabel("SPAM REAL")
  plt.ylabel("Total")


  # Treino e teste
  x = df.loc[:, ~df.columns.isin(["diagnosis"])]
  y = df["diagnosis"]

  # Treinamento do modelo
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 42)

  # Modelo regressão logistica
  from sklearn.linear_model import LogisticRegression
  
  model_regression_logistic = LogisticRegression()
  model_regression_logistic_fit = model_regression_logistic.fit(x_train, y_train)
  model_regression_logistic_scor = model_regression_logistic.score(x_train, y_train)
  print("Model - Logistic Regression: %.2f" % (model_regression_logistic_scor * 100))

  ## Previsão 
  model_regression_logistic_pred = model_regression_logistic.predict(x_test)
  model_regression_logistic_pred

  # Accuracy
  from sklearn.metrics import accuracy_score
  accuracy = accuracy_score(y_test, model_regression_logistic_pred)
  print("Acurácia - Logistic Regression: %.2f" % (accuracy * 100))

  # Roc
  roc_g = model_regression_logistic.predict_proba(x_test)[::,1]
  fpr, tpr, _ = metrics.roc_curve(y_test,  roc_g)
  auc = metrics.roc_auc_score(y_test, roc_g)
  plt.title("Curva roc - Regressão logistica")
  plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
  plt.legend(loc=4)
  plt.show()

  # Confusion matrix
  from sklearn.metrics import confusion_matrix
  matrix = confusion_matrix(y_test, model_regression_logistic_pred)
  matrix = sns.heatmap(matrix, cmap = 'YlGnBu', annot = True, fmt='g')

  # Classification report
  from sklearn.metrics import classification_report
  classification = classification_report(y_test, model_regression_logistic_pred)
  print("Modelo - Regressão logistica")
  print("\n")
  print(classification)

  # Métricas do modelo
  from sklearn.metrics import precision_score
  from sklearn.metrics import recall_score
  from sklearn.metrics import recall_score
  from sklearn.metrics import f1_score
  
  print("Modelo - Dummy Classifier")
  print("\n")
  print("Precision - Dummy Classifier = {}".format(precision_score(y_test, model_dummy_pred, average='macro')))
  print("Recall - Dummy Classifier = {}".format(recall_score(y_test, model_dummy_pred, average='macro')))
  print("Accuracy - Dummy Classifier = {}".format(accuracy_score(y_test, model_dummy_pred)))
  print("F1 Score - Dummy Classifier = {}".format(f1_score(y_test, model_dummy_pred, average='macro')))

  ####### Modelo hiperparâmetro

  # Validação cruzada
  from sklearn.model_selection import cross_validate
  
  results = cross_validate(model_regression_logistic, 
                         x_train, 
                         y_train, 
                         cv=5,
                         scoring=('accuracy'),
                         return_train_score=True)
  print(f"Mean train score {np.mean(results['train_score']):.2f}")
  print(f"Mean test score {np.mean(results['test_score']):.2f}")

 ## Hiperparâmetro GridSearchCV do modelo - Logistic Regression

  from sklearn.model_selection import GridSearchCV
  
  parametros = {
  "max_depth" : [3, 5],
  "min_samples_split" : [32, 64, 128],
  "min_samples_leaf" : [32, 64, 128],
  "criterion" : ["gini", "entropy"]
  }
  model_regression_logistic = LogisticRegression()
  DTCG = GridSearchCV(model_regression_logistic, parametros, cv = 5, return_train_score = True, scoring = "accuracy")
  grid_fit = DTCG.fit(x, y)
  results_GridSearchCV = grid_fit.cv_results_
  parametros = grid_fit.best_index_
  grid_pred = DTCG.predict(x_train)
  
  # Print do média treino, teste
  print(f"Mean train score {results_GridSearchCV['mean_train_score'][parametros]:.2f}")
  print(f"mean test score {results_GridSearchCV['mean_test_score'][parametros]:.2f}")
  results_GridSearchCV["params"][parametros]
  print("\n")
  
  # Acurácia do modelo
  acuracia = metrics.accuracy_score(y_train, grid_pred)
  print("Acuracia - GridSearchCV: %.2f" % (acuracia * 100))

  # Confusion matrix
  matrix = confusion_matrix(y_train, grid_pred)
  matrix = sns.heatmap(matrix, cmap = 'YlGnBu', annot = True, fmt='g')

  # Classification report
  print("GridSearchCV")
  print("\n")
  print(classification_report(y_train, grid_pred))

  # Métrica do hiperparametro GridSearchCV
  print("Modelo - GridSearchCV")
  print("Precision - Decision Tree = {}".format(precision_score(y_train, pred_randomized_search_cv, average='macro')))
  print("Recall - Decision Tree = {}".format(recall_score(y_train, pred_randomized_search_cv, average='macro')))
  print("Accuracy - Decision Tree = {}".format(accuracy_score(y_train, pred_randomized_search_cv)))
  print("F1 Score - Decision Tree = {}".format(f1_score(y_train, pred_randomized_search_cv, average='macro')))




```

## Melhorias

Que melhorias você fez no seu código? 
- Ex: refatorações, melhorias de performance, acessibilidade, etc


## Suporte

Para suporte, mande um email para rafaelhenriquegallo@gmail.com

