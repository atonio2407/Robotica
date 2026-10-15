## Avance Proyecto Robótica

Asignatura: Robótica e IA.

Profesor: Nazhir Jesús Amaya Tejera.

Estudiantes: Adrián Martínez, Jesús Hernández, Yeiner Castro.

Fase: Avances del proyecto del 2do corte.

## 1. Título, Objetivo General y Objetivos Específicos

Título del Proyecto: Sistema Automatizado de Clasificación y Conteo de Tuercas mediante Visión Artificial y Brazo Robótico Dobot Magician.

Objetivo General: Desarrollar un sistema robótico que, mediante visión artificial, cuente las tuercas presentes sobre una cinta transportadora y retire las sobrantes con el brazo Dobot Magician, de modo que solo pase la cantidad x de tuercas definida por el usuario.

## Objetivos Específicos:

- 1. Entrenar un modelo de detección de objetos utilizando la arquitectura YOLOv8n mediante la plataforma Google Colab y un conjunto de datos extraídos de Roboflow.

- 2. Integrar el modelo de visión artificial con el entorno de control del brazo robótico Dobot Magician y una cámara USB para garantizar la detección en tiempo real de las piezas.

- 3. Programar la lógica de control cinemático y de flujo para que el brazo robótico retire las tuercas sobrantes de la cinta transportadora una vez se haya alcanzado la cantidad objetivo establecida.

## 2. Planteamiento del Problema

En los entornos industriales modernos, la clasificación, el conteo y la separación de piezas pequeñas, como las tuercas, son tareas fundamentales pero altamente repetitivas. Cuando estos procesos se realizan de manera manual, están sujetos a errores humanos, fatiga y tiempos de ejecución prolongados, lo que disminuye la eficiencia general de la cadena de producción. El problema radica en la necesidad de automatizar el conteo exacto de un lote de piezas en movimiento sin interrumpir el flujo constante de la línea de ensamblaje. Para resolver esto, el presente proyecto plantea la implementación de una estación de clasificación automatizada a escala utilizando un brazo robótico Dobot Magician, el cual, guiado por un sistema de visión artificial, sea capaz de discernir cuándo se ha cumplido una cuota específica de tuercas y actuar mecánicamente para retirar el excedente, garantizando un control de calidad preciso y continuo.


## 3. Estado del Arte y Marco Teórico

3.1. Estado del Arte: La automatización mediante robótica guiada por visión (VGR, por sus siglas en inglés) es una tendencia consolidada en la Industria 4.0. Investigaciones recientes destacan el uso de redes neuronales convolucionales (CNN), como las diferentes versiones de YOLO (You Only Look Once), para la detección de piezas en bandas transportadoras con alta precisión y baja latencia. Proyectos similares a nivel industrial utilizan brazos SCARA o antropomórficos para el "pick and place" (recoger y colocar) inteligente, adaptando sus trayectorias en tiempo real según la retroalimentación visual. Nuestro proyecto escala este concepto industrial a un entorno de laboratorio utilizando el ecosistema Dobot.

## 3.2. Marco Teórico

- Brazo Robótico Dobot Magician: Es un brazo robótico de escritorio de 4 ejes diseñado para la educación e investigación. Posee alta repetibilidad y es capaz de realizar tareas de agarre, succión e impresión 3D.

- Visión Artificial y YOLOv8n: La visión artificial permite a las computadoras extraer información significativa de imágenes digitales. YOLOv8 es una arquitectura de detección de objetos de última generación, conocida por su equilibrio entre velocidad y precisión. La versión "n" (nano) es la más ligera, ideal para ejecutarse con rapidez y procesar el flujo de video en tiempo real de la cámara USB.

- Google Colab y Roboflow: Colab proporciona un entorno de cuadernos (notebooks) basados en la nube que permite la ejecución de código en Python con acceso a GPUs para acelerar el entrenamiento de redes neuronales. Roboflow es una plataforma que facilita el alojamiento, anotación y preprocesamiento de conjuntos de datos de imágenes, fundamental para preparar las imágenes de las tuercas antes del entrenamiento.

## 4. Introducción y Lista de Materiales

4.1. Introducción El presente documento describe los avances correspondientes al segundo corte de la asignatura Robótica e IA. El proyecto busca fusionar los conceptos de cinemática robótica y aprendizaje automático para crear una solución práctica: un organizador inteligente de tuercas. A través de la programación, el sistema identifica visualmente las piezas que transitan por una cinta transportadora, lleva un registro numérico y, mediante la acción física del robot Dobot Magician, gestiona el flujo de material retirando las piezas que superen la cantidad deseada. Este desarrollo demuestra la aplicabilidad de los conocimientos adquiridos en clase en la resolución de problemas de automatización logística.


## 4.2. Lista de Materiales

- 1. Brazo Robótico Dobot Magician.

- 2. Cinta Transportadora Dobot (Mini Conveyor Belt Kit).

- 3. Cámara Web USB.

- 4. Computadora con software de control (Python / DobotStudio).

- 5. Tuercas metálicas de prueba.

- 6. Entorno de desarrollo Google Colab (Software).

- 7. Conjunto de datos de Roboflow (Nuts dataset).

*Entorno fisico (tuercas y cinta transportadora)*

*Entrenamiento en Python/Colab. Uso de YOLOv8n*

*Sistema de actuacion Dobot Magician*

*Cémara USB*

*Captura de fotogramas en*

*tiempo real*

*Ilustración 1. Componentes del proyecto.*

## 5. Desarrollo del Componente Técnico

El desarrollo del sistema se divide en dos grandes bloques: la percepción (Inteligencia Artificial) y la actuación (Robótica).

5.1. Entrenamiento del Modelo de Visión Se utilizó el conjunto de datos de dominio público "nuts" disponible en Roboflow. Las imágenes fueron preprocesadas y exportadas en formato YOLO. Posteriormente, se utilizó un entorno de Google Colab con aceleración por hardware (GPU) para entrenar el modelo YOLOv8n. El entrenamiento se ajustó para reconocer exclusivamente la clase correspondiente a las tuercas, generando un archivo de pesos (weights) optimizado para inferencia en tiempo real.


5.2. Lógica de Control e Integración La cámara USB se posiciona cenitalmente sobre la cinta transportadora para monitorear el área de trabajo del robot. El flujo del programa es el siguiente:

- 1. La cámara captura el flujo de video y el modelo YOLOv8n procesa cada fotograma.

- 2. El sistema lleva un contador de las tuercas detectadas que cruzan una línea virtual en la pantalla.

- 3. Si la cantidad de tuercas detectadas es menor o igual a la variable objetivo (x), la cinta sigue avanzando.

- 4. Si la cámara detecta tuercas que superan la cantidad x, el sistema calcula las coordenadas espaciales de la tuerca sobrante, pausa la cinta transportadora temporalmente, y envía los comandos cinemáticos (X, Y, Z) al Dobot Magician.

- 5. El robot agarra las tuercas excedentes y las traslada a un área de descarte y retorna a su posición inicial (Home).

*Ilustración 2. Diagrama de bloques de lógica de control del sistema.*


## 6. Verificación de Metodología, Cronograma y Dificultades

6.1. Metodología: Se ha empleado una metodología iterativa e incremental. La primera fase consistió en aislar la variable de visión artificial (lograr que el modelo detecte de forma confiable). La segunda fase se centró en la calibración espacial del robot. La tercera fase, actualmente en curso, es la integración concurrente de ambos sistemas para operar en tiempo real.

## 6.2. Cronograma de Actividades

| Fase | Actividad | Estado |
| --- | --- | --- |
| 1 | Adquisición de dataset en Roboflow y preparación. | Completado |
| 2 | Entrenamiento del modelo YOLOv8n en Google Colab. | Completado |
| 3 | Montaje de Hardware (Dobot, Cinta y Cámara USB). | Completado |
| 4 | Programación de rutinas de movimiento (Pick and Place). | En progreso |
| 5 | Integración de visión y control cinemático. | En progreso |
| 6 | Pruebas de funcionamiento y corrección de errores. | Pendiente |

6.3. Dificultades Encontradas: Durante el desarrollo se han identificado los siguientes retos técnicos:

- 1. Condiciones de Iluminación: La cámara USB es sensible a los cambios de luz ambiental, lo que inicialmente generaba falsos positivos o falta de detección en el modelo YOLOv8n. Se requirió estandarizar la iluminación del entorno de pruebas.

- 2. Calibración de Coordenadas: Transformar las coordenadas de los píxeles de la imagen de la cámara (2D) a las coordenadas espaciales del brazo robótico Dobot Magician (3D) requiere una matriz de transformación precisa para evitar errores milimétricos en el agarre de la tuerca.
