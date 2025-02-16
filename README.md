# MNIST Digit Recognition with CNN

**Python** Versión requerida: 3.8+  
**TensorFlow** Versión requerida: 2.12+  
**Licencia** MIT  

Proyecto para clasificar dígitos manuscritos usando una red neuronal convolucional (CNN) entrenada con el dataset MNIST.

## Instalación

Clona el repositorio:
```bash
git clone https://github.com/3thanF/mnist-digit-recognition.git
```

## Instala dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del proyecto

- 📁 mnist-digit-recognition/
  - 📁 data/
  - 📁 notebooks/
    - 📄 MNIST_Exploracion.ipynb
  - 📁 src/
    - 📄 train_model.py
    - 📄 predict.py
  - 📁 models/
  - 📁 reports/
  - 📄 requirements.txt
  - 📄 README.md

## Cómo entrenar el modelo

Ejecuta el script de entrenamiento:
``` bash
python src/train\_model.py
```

## Ejemplo de predicción

Usa una imagen de ejemplo para probar el modelo:
``` bash
python src/predict.py --image\_path samples/ejemplo\_5.png
```

## Resultados

Precisión en test: 99.2%

Matriz de confusión disponible en: reports/confusion\_matrix.png

## Recursos

Dataset MNIST en Keras: https://keras.io/api/datasets/mnist/

Tutorial de CNN: [Enlace a tutorial relevante]

## Licencia

MIT License

Copyright (c) 2025 Ethan Fallas

Se concede permiso, libre de cargos, a cualquier persona que obtenga una copia de este software y los archivos de documentación asociados (el "Software"), a utilizar el Software sin restricción, incluyendo sin limitación los derechos a usar, copiar, modificar, fusionar, publicar, distribuir, sublicenciar y/o vender copias del Software.