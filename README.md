# TD3
Algoritmo de aprendizaje por refuerzo RPG-TD3 para entrtenar el humanoide de gym.



El código se estructura con una librería general 'TD3.py' y difernetes notebook's con diferentes versiones de entrenamiento para el agente.

- En la librería 'TD3.py' es donde se ha implementado mis versiones del Actor y los Críticos así como el agente TD3. Además en esta librería
mantengo las funciones que se han utilizado en el bucle general del entrenamiento así como la generación del entorno de ejecución y el control de los ficheros y graficas de 
resultados.

(*) Las diferentes versiones del Notebook's  marcan las diferentes estrategias que se han dado para la Explotacion Vs Exploración: Notar que en todas las versiones 
a las acciones seleccionadas tanto en exploración como en explotación se le añade un ruido que permite una variabilidad de las acciones que permite seguir explotando.

 - V0_1 -->  Esstrategia AdaptiveHybridPolicy:


	Exploración:

		Durante los primeros pasos (total_steps < start_steps), el agente explora el entorno tomando acciones al azar.
		Luego de estos pasos iniciales, se reduce gradualmente la probabilidad de exploración para comenzar a confiar más en el conocimiento adquirido hasta el momento.

	Explotación:

		Después de superar los primeros pasos, el agente comienza a utilizar su política aprendida con una probabilidad. Sin embargo, aún se mantiene una pequeña probabilidad de exploración 
		para evitar el estancamiento y permitir una mejora continua de la política. Durante el entrenamiento, se registran las recompensas, las pérdidas y el factor de exploración promedio 
		para cada episodio. Estos datos se utilizan para analizar y visualizar el rendimiento del agente a lo largo del tiempo.

 -  V1_1 --> Estrategia AdaptiveBalancedPolicy: 2 fases diferenciadas, una de explotaración en exclusiva y otra de explotación + ruido 
 -  V2_1 --> Estrategia EGreedyPolicy: Estrategia E-greedy en la fase de expplotación con una probabiidad fija para seguir explorando. Aparte de la adición de ruido en las dos fases
 -  v3_1 --> Modificamos la estrategia de Exploracion Vs Explotción. Haremos un entrenamiento en dos partes.

		1 ) Recorremos el entorno en busca de observaciones y acciones asociadas subóptmas. Rellenamos la memoria de repetición con estas observciones.

		2) Entrenamos de forma normal tal que cuando vayamos a seleccionar la acción tenemos varias fases.:

				a ) primera fase donde usamos las acciones almacenadas en el replaybuffer.

				b ) segunda fase de esplotación de las acciones de la política pero con una probabilidad de realizar mñas explotaciones, tal que eta probabilidad va en decaimiento a medida que hacemos más pasos.

      


(*) Se ha implementado diferentes opciones del RaplayBuffer:

 - Simple: Se mantiene un buffer simple de transiciones con capacidad maxima donde cuando se llega a su capacidad máxima se almacena al principio. Es la mejor opción y la que
  al final se ha mntenido en las versiones   V0_1, V1_1 y V2_1.
 - SimplePrioritizedReplayBuffer: Se almacena con un peso dado pro el error loss que nos devuelve el cñritico (calidad de la plítica). 
 - PrioritizedReplayBuffer: se almacena con un peso que será la prioridad. Est epeso se calcula por una exonencial del loos del crítico. El problema es que debe actualizarse en cada inserción
lo que genera un aumento del costo computacional que no justifica el uso dado los resultados que se iban dando. 
 - ReplayBufferPos: sólo se lamacena las transiciones con recompensas positivas y luego se buscan aquellas acciones que se ejecutan en un estado mñs cercano al actual. Sñolo sobreajusta 
tanto el crítico como el actor. Son los  peores resultados a ceñirse exclusivamente a unas pocaa transiciones.

 Tanto con SimplePrioritizedReplayBuffer como con PrioritizedReplayBuffer no hay datos porque al final pare l ejcución. Son ejecuciones que tardan días en reproducirse y cuando veía qu een una tarde 
sólo se ejecutaban 5mil pasos cuando lo normal es tener d 1 millon de pasos en adelante, no tenía sentido. Ademañs se iba viendo que tanto el loss del crítico como del actor iban en crecimiento.


