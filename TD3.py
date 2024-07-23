'''
Titulo:TRABAJO AP. DDPG TD(3) (Gradiente de política determinista profunda (TD3) de doble retardo. Entornos de estados continuos: Humanoide). + Conexión Chatgpt (o similar) + modelo dalle (similr)
Author: José Javier Gutiérrez Gil
Date: 2024-02-18
Univeridad de Valencia. Grado de Ciencia de Datos


# Pasos del código:

1.  Construir la **memoria de la repetición de experiencias**

2.  Construir una red neuronal para el **actor del modelo** y una red neuronal para el **actor del objetivo (Target)**

3. Construir dos redes neuronales para los dos **críticos del modelo** y dos redes neuronales para los dos ***críticos del objetivo (Target)**

4. Construir **clase TD3** que contendra el **entrenamiento del modelo**

5. **Inicializamos** los **hiperparámetros**

6. **Creamos las carpetas y nombres de fichero** donde se almacenarán los modelos y su entrenamiento.

7. **Cargamos el entorno deseado de gym** y obtenemos información de sus estados y acciones

8. **Entrenamos el entorno** Humanoide  con el DDPG TD3

9. **Leemos los ficheros almacenados en disco** de los modelos y el entrenamient, y **creamos los videos del entrenamiento**

'''
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from collections import deque


# Selección del dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
Item 1: Construir la memoria de la repetición de experiencia-

Se ha creado varias propuestas, pero la ejor opcion, para evitar arga computacional es la ReplayBuffer estandar  donde
se almacena de forma cíclicla las transiciones.
'''
 ##################
## Clase que simula la memoria de experiencias de repetición
## Esta es la primera implementación utilizada de la memoria de repetición empleada en el algoritmo de entrenamiento
####################
class ReplayBuffer (object):
    """
      Clase gençerica para almacenar y muestrear transiciones para el entrenamiento de un modelo en un algoritmo de RL.

      Args:
          max_capacity (float): Tamaño máximo del búfer de repetición (por defecto, 1e6).
    """

    def __init__ (self, max_capacity = 1e10):
        """
          Inicializa la clase ReplayBuffer.

          Args:
              max_capacity (float): Tamaño máximo del búfer de repetición (por defecto, 1e6).
        """
        self.storage      = []             # Almacenar las transiciones de la memoria de repetición, simulando la experiencia pasada del actor durante el entrenamiento.
        self.max_capacity = max_capacity   # Capacidad máxima de transiciones que el búfer de repetición puede almacenar
        self.pointer      = 0              # Ìndice que indica qué transición se utilizará a continuación en el búfer de repetición durante el proceso de muestreo.

    def add (self, transition):
        """
          Agrega una transición al búfer de repetición.

          Args:
              transition (tuple): Una tupla que representa una transición (estado, próximo estado, acción, recompensa, done).
        """
        if len(self.storage) == self.max_capacity:
            self.storage[int(self.pointer)] = transition
            self.pointer = (self.pointer + 1) % self.max_capacity
        else:
            self.storage.append(transition)

    def sample (self, batch_size):
        """
          Muestrea un lote de transiciones del búfer de repetición.

          Args:
              batch_size (int): Tamaño del lote de transiciones a muestrear.

          Returns:
              batch_states (np.array):      Matriz de estados del lote de transiciones.
              batch_next_states (np.array): Matriz de próximos estados del lote de transiciones.
              batch_actions (np.array):     Matriz de acciones del lote de transiciones.
              batch_rewards (np.array):     Matriz de recompensas del lote de transiciones.
              batch_dones (np.array):       Matriz de terminaciones del lote de transiciones.
        """
        indices = np.random.randint (0, len (self.storage), size = batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in indices:
            state, next_state, action, reward, done = self.storage [i]
            batch_states.append (np.array (state, copy = False))
            batch_next_states.append (np.array (next_state, copy = False))
            batch_actions.append (np.array (action, copy = False))
            batch_rewards.append (np.array (reward, copy = False))
            batch_dones.append (np.array (done, copy = False))

        return (
            np.array (batch_states),
            np.array (batch_next_states),
            np.array (batch_actions),
            np.array (batch_rewards).reshape (-1, 1),
            np.array (batch_dones).reshape (-1, 1)
        )

        
##################
## Clase que simula la memoria de experiencias de repetición, sólo con recompensa positiva: class ReplayBufferPos (object):

# En el método add(), primero se obtiene la recompensa de la nueva transición.
# Luego, se compara esta recompensa con las recompensas de las transiciones ya almacenadas en el búfer. Si la nueva recompensa es mayor que todas # las recompensas ya almacenadas, la transición se añade al búfer.
# Si el tamaño del búfer supera la capacidad máxima después de añadir la nueva transición, se elimina la transición con la recompensa más baja.
# De lo contrario, la nueva transición no se añade al búfer. Esto asegura que solo se almacenen las transiciones con las recompensas más altas.
####################

class ReplayBufferPos(object):
    """
    Clase genérica para almacenar y muestrear transiciones para el entrenamiento de un modelo en un algoritmo de RL.

    Args:
        max_capacity (int): Tamaño máximo del búfer de repetición (por defecto, 1e4).
    """

    def __init__(self, max_capacity=1e4):
        """
        Inicializa la memoria de repetición.

        Args:
            max_capacity (int): Capacidad máxima de la memoria de repetición.
        """
        self.max_capacity = max_capacity
        self.pointer = 0
        self.storage = []  # Lista para almacenar las transiciones

    def add(self, transition):
        """
          Agrega una transición al búfer de repetición.

          Args:
              transition (tuple): Una tupla que representa una transición (estado, próximo estado, acción, recompensa, done).
        """
        if len(self.storage) == self.max_capacity:
            self.storage[int(self.pointer)] = transition
            self.pointer = (self.pointer + 1) % self.max_capacity
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        """
        Muestrea un lote de transiciones de la memoria de repetición.

        Args:
            batch_size (int): Tamaño del lote de transiciones a muestrear.

        Returns:
            Tuple: Lote de estados, próximos estados, acciones, recompensas y señales de terminación.
        """
        if not self.storage:
            return None

        # Seleccionar índices aleatorios para muestrear un lote de transiciones.
        indices = np.random.choice(len(self.storage), size=batch_size, replace=False)
        # Extraer columnas relevantes de las transiciones seleccionadas.
        batch_states      = np.array([self.storage[i][0] for i in indices])
        batch_next_states = np.array([self.storage[i][1] for i in indices])
        batch_actions     = np.array([self.storage[i][2] for i in indices])
        batch_rewards     = np.array([self.storage[i][3] for i in indices]).reshape(-1, 1)
        batch_dones       = np.array([self.storage[i][4] for i in indices]).reshape(-1, 1)

        return batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones

    def sample_action(self, obs):
        """
        Selecciona una acción basada en el estado más cercano en la memoria de repetición.

        Args:
            obs (np.array): Estado actual.

        Returns:
            Tuple: Lote de estados, próximos estados, acciones, recompensas y señales de terminación correspondientes a la transición más cercana.
        """
        if not self.storage:
            return None

        # Calcular la distancia entre el estado actual y todos los estados en la memoria de repetición.
        distances          = [np.linalg.norm(obs - transition[0]) for transition in self.storage]
        closest_index      = np.argmin(distances)
        closest_transition = self.storage[closest_index]

        # Extraer los elementos de la transición más cercana.
        batch_states      = np.array([closest_transition[0]])
        batch_next_states = np.array([closest_transition[1]])
        batch_actions     = np.array([closest_transition[2]])
        batch_rewards     = np.array([closest_transition[3]]).reshape(-1, 1)
        batch_dones       = np.array([closest_transition[4]]).reshape(-1, 1)

        return batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones
####################################################
#### SimplePrioritizedReplayBuffer:
###################################################
## La memoria de repetición con prioridad simple tiene algunas ventajas y desventajas
## en comparación con la memoria de repetición aleatoria:
## Ventajas:
##    - Priorización de transiciones importantes: La memoria de repetición con
##      prioridad simple asigna prioridades a las transiciones según su error de
##      predicción, lo que permite que las transiciones más importantes se muestreen
##      con mayor frecuencia.
##    - Eficiencia en el uso de memoria: Al priorizar las transiciones más importantes,
##      se puede lograr una mejor eficiencia en el uso de la memoria, ya que las transiciones
##      menos importantes pueden ser eliminadas o muestreadas con menos frecuencia.
## Desventajas:
##   - Sensibilidad a errores de estimación:  Está muy relacionado al cálculo del error ya que la prioridad
##     se basa en dicho error.
##   - Mayor complejidad: Comparada con la memoria de repetición aleatoria, la implementación de las operaciones
##     para la priorización puede aumentar la carga computacional y los resutados obtenidos no justifiquen dicho
##     aumento de la carga.
##   - Sesgo de priorización: La priorización puede introducir sesgos en el proceso de muestreo,
##       lo que puede llevar a una exploración subóptima o a un aprendizaje inestable si no se maneja adecuadamente.
##       (puede que si los errores son muy proximos siempre se muestreen las mismas transicones y no salgas del local)
##
##  Nota: epleada cuando el entrenamiento lo tenía tal que se evaluava cada paso . Con la nueva implementación no se puede
##      aplicar directamente porque no tiene priporidades basadas en los errores ya que no se ha evaluado el paso. ---123--
##

class SimplePrioritizedReplayBuffer:
    """
    Clase que implementa un búfer de repetición con prioridad simple para el aprendizaje por refuerzo.

    Args:
        max_capacity (int): Capacidad máxima del búfer de repetición.
        epsilon (float)   : Valor pequeño añadido a los errores de predicción para evitar que sean cero.

    Attributes:
        max_capacity (int): Capacidad máxima del búfer de repetición.
        epsilon (float)   : Valor pequeño añadido a los errores de predicción para evitar que sean cero.
        storage (list)    : Lista que almacena las transiciones del búfer de repetición.
        pointer (int)     : Índice que indica qué transición se utilizará a continuación en el búfer de repetición durante el proceso de muestreo.
    """

    def __init__(self, max_capacity=1e10, epsilon=1e-4):
        self.storage = []
        self.max_capacity = max_capacity
        self.pointer = 0
        self.epsilon = epsilon

    def add(self, error, transition):
        """
              Agrega una transición al búfer de repetición.

              Args:
                  error (float)      : Error de predicción del modelo asociado a la transición.
                  transition (tuple) : Una tupla que representa una transición (estado, próximo estado, acción, recompensa, done).
        """
        state, next_state, action, reward, done = transition

        # Convertir los elementos de la transición a arrays NumPy si es necesario
        state = np.array(state)
        next_state = np.array(next_state)
        action = np.array(action)

        # Convertir la recompensa y la bandera done a arrays de un solo elemento
        reward = np.array([reward], dtype='float32') if isinstance(reward, (float, int)) else np.array(reward)
        done = np.array([done], dtype='float32') if isinstance(done, (float, int)) else np.array(done)

        priority = min(error, 1.0) + self.epsilon  # Se utiliza el mínimo entre el error y 1.0 como prioridad
        if len(self.storage) == self.max_capacity:
            min_priority = min(p for p, _ in self.storage)  # Calcula la prioridad mínima actual
            for i, (p, _) in enumerate(self.storage):
                if p == min_priority:
                    # Reemplaza la transición con la prioridad mínima con la nueva transición
                    self.storage[i] = (priority, (state, next_state, action, reward, done))
                    break
        else:
            self.storage.append((priority, (state, next_state, action, reward, done)))
    def sample(self, batch_size):
        """
        Muestrea un lote de transiciones del búfer de repetición.

        Args:
            batch_size (int): Tamaño del lote de transiciones a muestrear.

        Returns:
            batch_states (np.array)      : Matriz de estados del lote de transiciones.
            batch_next_states (np.array) : Matriz de próximos estados del lote de transiciones.
            batch_actions (np.array)     : Matriz de acciones del lote de transiciones.
            batch_rewards (np.array)     : Matriz de recompensas del lote de transiciones.
            batch_dones (np.array)       : Matriz de terminaciones del lote de transiciones.
        """
        if len(self.storage) < batch_size:
            num_to_sample = batch_size - len(self.storage)
            indices = np.random.randint(0, len(self.storage), size = num_to_sample)
            indices = np.concatenate((np.arange(len(self.storage)), indices))
        else:
            priorities = np.array([priority.detach().numpy() if torch.is_tensor(priority) else priority for priority, _ in self.storage])


            # priorities = np.array([priority for priority, _ in self.storage])
            probs = priorities / priorities.sum()  # Probabilidades de muestreo
            indices = np.random.choice(len(self.storage), size = batch_size, p = probs)

        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in indices:
            priority, transition = self.storage [i]

            state, next_state, action, reward, done = transition

            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))


        return (
            np.array(batch_states),
            np.array(batch_next_states),
            np.array(batch_actions),
            np.array(batch_rewards).reshape(-1, 1),
            np.array(batch_dones).reshape(-1, 1)
        )

    def add_transitions_to_replay_buffer(self, error, state, next_state, action, reward, done, batch_size):
        """
        Agrega transiciones al búfer de repetición.

        Args:
            error (float): Error para el conjunto de acciones batch.
            state (list): Lista de estados.
            next_state (list): Lista de estados siguientes.
            action (list): Lista de acciones.
            reward (list): Lista de recompensas.
            done (list): Lista de indicadores de finalización.
            batch_size (int): Tamaño del lote.

        Returns:
            None
        """

        for i in range (batch_size):
            self.add (error, (state[i], next_state[i], action[i], reward[i], done[i]))
####################################################
#### PrioritizedReplayBuffer:
###
#    - Esta implementación prioriza las transiciones en función de su importancia
#    relativa, lo que significa que las transiciones más significativas tienen
#    una mayor probabilidad de ser muestreadas.
#    - se utiliza un enfoque simple para calcular las prioridades basadas en los errores de predicción del modelo
#    Utiliza un esquema de muestreo ponderado basado en las prioridades de las
#    transiciones, lo que aumenta la probabilidad de seleccionar transiciones importantes.
#    La función update_priorities actualiza las prioridades de las transiciones en
#    función de los errores de predicción del modelo, lo que permite adaptar la
#    importancia de las transiciones a medida que avanza el entrenamiento.
#########
# En teoria esta implementación debe de mejorar sustancialmente el entrenamiento:
#  - Mayor eficiencia de entrenamiento:
#     Al priorizar las transiciones importantes, se enfoca en aprender de las experiencias
#     más relevantes, lo que puede llevar a una mejora en la eficiencia del entrenamiento
#     y una convergencia más rápida del modelo.
#  - Mejora en el aprendizaje:
#     Al asignar una mayor probabilidad a las transiciones importantes, el modelo
#     puede aprender de manera más efectiva, especialmente en situaciones donde ciertas
#     transiciones son críticas para el aprendizaje.
#  - Adaptabilidad a cambios en la importancia de las transiciones:
#     La capacidad de actualizar dinámicamente las prioridades de las transiciones
#     permite que el modelo se adapte a cambios en la importancia de las experiencias a
#     medida que progresa el entrenamiento, lo que resulta en un aprendizaje más robusto y adaptativo.
#

class PrioritizedReplayBuffer:
    """
    Clase que implementa un búfer de repetición con prioridad para el aprendizaje por refuerzo.

    Args:
        max_capacity (int): Capacidad máxima del búfer de repetición.
        alpha (float): Exponente que controla cómo se ajustan las probabilidades de muestreo.
        beta (float): Exponente que controla cómo se pesan los errores de la red.
        epsilon (float): Valor pequeño añadido a las prioridades para evitar que sean cero.

    Attributes:
        max_capacity (int): Capacidad máxima del búfer de repetición.
        alpha (float): Exponente que controla cómo se ajustan las probabilidades de muestreo.
        beta (float): Exponente que controla cómo se pesan los errores de la red.
        epsilon (float): Valor pequeño añadido a las prioridades para evitar que sean cero.
        buffer (list): Lista que almacena las transiciones del búfer de repetición.
        priorities (np.array): Array que almacena las prioridades de las transiciones.
        pos (int): Índice que indica la posición actual en el búfer de repetición.
        max_priority (float): Prioridad máxima en el búfer de repetición.
    """

    def __init__ (self, max_capacity, alpha = 0.6, beta = 0.4, epsilon = 1e-4):
        self.max_capacity = max_capacity
        self.alpha        = alpha
        self.beta         = beta
        self.epsilon      = epsilon
        self.buffer       = []
        self.priorities   = np.zeros(max_capacity, dtype=np.float32)
        self.pos          = 0
        self.max_priority = 1.0

    def add (self, transition):
        """
        Añade una transición al búfer de repetición.

        Args:
            transition: Una tupla que representa una transición (estado, próximo estado, acción, recompensa, done).
        """
        max_priority = self.priorities.max () if self.buffer else 1.0
        if len(self.buffer) < self.max_capacity:
            self.buffer.append (transition)
        else:
            self.buffer [self.pos] = transition
        self.priorities [self.pos] = max_priority
        self.pos = (self.pos + 1) % self.max_capacity

        # Después de agregar la transición, se actualizan las prioridades
        self.update_priorities ([self.pos], [self.max_priority])

    def sample (self, batch_size):
        """
        Realiza un muestreo de transiciones del búfer de repetición.

        Args:
            batch_size (int): Tamaño del lote de transiciones a muestrear.

        Returns:
            batch_states (np.array):      Matriz de estados del lote de transiciones.
            batch_next_states (np.array): Matriz de próximos estados del lote de transiciones.
            batch_actions (np.array):     Matriz de acciones del lote de transiciones.
            batch_rewards (np.array):     Matriz de recompensas del lote de transiciones.
            batch_dones (np.array):       Matriz de terminaciones del lote de transiciones.
            indices (np.array):           Índices de las transiciones muestreadas.
            weights (np.array):           Pesos de importancia de las transiciones muestreadas.
        """
        priorities = self.priorities[:len(self.buffer)]
        probs      = priorities ** self.alpha
        probs     /= probs.sum ()
        indices    = np.random.choice (len (self.buffer), batch_size, p = probs)
        samples    = [self.buffer [idx] for idx in indices]
        total      = len(self.buffer)
        weights    = (total * probs[indices]) ** (-self.beta)
        weights   /= weights.max ()

        # Convertir las transiciones muestreadas en matrices numpy separadas
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for transition in samples:
            state, next_state, action, reward, done = transition
            batch_states.append (state)
            batch_next_states.append (next_state)
            batch_actions.append (action)
            batch_rewards.append (reward)
            batch_dones.append (done)

        return (
            np.array (batch_states),
            np.array (batch_next_states),
            np.array (batch_actions),
            np.array (batch_rewards).reshape(-1, 1),
            np.array (batch_dones).reshape(-1, 1),
            indices,
            np.array (weights, dtype=np.float32)
        )
    def update_priorities (self, indices, errors):
        """
        Actualiza las prioridades de las transiciones en el búfer de repetición.

        Args:
            indices (list)   : Índices de las transiciones cuyas prioridades se van a actualizar.
            errors (np.array): Errores de predicción del modelo asociados a las transiciones.

        """
        priorities = np.abs(errors) + self.epsilon
        for idx, priority in zip(indices, priorities):
            self.priorities [idx] = priority
            self.max_priority = max (self.max_priority, priority)

    def update_priorities_from_train (self, indices, errors, weights):
        """
        Actualiza las prioridades de las transiciones durante el proceso de entrenamiento.

        Args:
            indices (list)    : Índices de las transiciones cuyas prioridades se van a actualizar.
            errors (np.array) : Errores de predicción del modelo asociados a las transiciones.
            weights (np.array): Pesos de importancia de las transiciones.
        """


        priorities = np.abs(errors * weights) + self.epsilon
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


'''
## **Item 2:** Construir una clase Actor que definirá tanto el **actor del modelo** y el **actor del objetivo (Target)**. (Los dos tendrán la mima arquitectura de NN)
'''
class Actor (nn.Module):
    """
      Clase que define la arquitectura de la red neuronal del actor en un algoritmo de aprendizaje por refuerzo.

      Args:
          state_dimensions (int):   Dimensión del espacio de estados de entrada.
          action_dimensions (int):  Dimensión del espacio de acciones de salida.
          max_action_value (float): Valor máximo permitido para las acciones.
    """

    def __init__ (self, state_dimensions, action_dimensions, max_action_value):
        """
          Inicializa la clase Actor.

          Args:
              state_dimensions (int):   Dimensión del espacio de estados de entrada.
              action_dimensions (int):  Dimensión del espacio de acciones de salida.
              max_action_value (float): Valor máximo permitido para las acciones.
        """
        super (Actor, self).__init__ ()
        self.layer_1 = nn.Linear (state_dimensions, 400) #400
        self.layer_2 = nn.Linear (400, 300) #400,300
        self.layer_3 = nn.Linear (300, action_dimensions) #300
        self.max_action_value = max_action_value

    def forward (self, x):
        """
          Método que define el pase hacia adelante (forward pass) de la red neuronal del actor.

          Args:
              x (torch.Tensor): Tensor de entrada representando el estado actual.

          Returns:
              action (torch.Tensor): Tensor de salida representando la acción elegida por el actor.
        """
        x = F.relu (self.layer_1(x))
        x = F.relu (self.layer_2(x))
        action = self.max_action_value * torch.tanh (self.layer_3(x))
        return action

'''
Item 3: Construir la clase Critico. La misma arquitectura de NN la conformarán los dos críticos del modelo y los dos críticos del objetivo (Target)
'''
class Critic (nn.Module):
    """
      Clase que define la arquitectura de la red neuronal del crítico en un algoritmo de aprendizaje por refuerzo.

      Args:
          state_dimensions (int):  Dimensión del espacio de estados de entrada.
          action_dimensions (int): Dimensión del espacio de acciones de entrada.
    """

    def __init__ (self, state_dimensions, action_dimensions):
        """
          Inicializa la clase Critic.

          Args:
              state_dimensions (int): Dimensión del espacio de estados de entrada.
              action_dimensions (int): Dimensión del espacio de acciones de entrada.
        """
        super (Critic, self).__init__()
        # Primer crítico como red neuronal profunda
        self.layer_1_1 = nn.Linear (state_dimensions + action_dimensions, 400) #400
        self.layer_2_1 = nn.Linear (400,300)#400,300
        self.layer_3_1 = nn.Linear (300, 1) #300
        # Segundo crítico como red neuronal profunda
        self.layer_1_2 = nn.Linear (state_dimensions + action_dimensions, 400)
        self.layer_2_2 = nn.Linear (400, 300)#400,300
        self.layer_3_2 = nn.Linear (300, 1)#300

    def forward (self, state, action):
        """
          Método que define el pase hacia adelante (forward pass) de la red neuronal del crítico.

          Args:
              state (torch.Tensor): Tensor de entrada representando el estado actual.
              action (torch.Tensor): Tensor de entrada representando la acción elegida por el actor.

          Returns:
              Q1 (torch.Tensor): Valor de la función de valor (Q-value) estimado por el primer crítico.
              Q2 (torch.Tensor): Valor de la función de valor (Q-value) estimado por el segundo crítico.
        """
        xu = torch.cat ([state, action], 1)
        # Propagación hacia adelante del primer crítico
        x1 = F.relu (self.layer_1_1(xu))
        x1 = F.relu (self.layer_2_1(x1))
        Q1 = self.layer_3_1 (x1)
        # Propagación hacia adelante del segundo crítico
        x2 = F.relu (self.layer_1_2(xu))
        x2 = F.relu (self.layer_2_2(x2))
        Q2 = self.layer_3_2 (x2)
        return Q1, Q2

    def Q1 (self, state, action):
        """
        Método que calcula el valor de la función de valor (Q-value) estimado por el primer crítico.

        Args:
            state (torch.Tensor): Tensor de entrada representando el estado actual.
            action (torch.Tensor): Tensor de entrada representando la acción elegida por el actor.

        Returns:
            Q1 (torch.Tensor): Valor de la función de valor (Q-value) estimado por el primer crítico.
        """
        xu = torch.cat ([state, action], 1)
        x1 = F.relu (self.layer_1_1 (xu))
        x1 = F.relu (self.layer_2_1 (x1))
        Q1 = self.layer_3_1 (x1)
        return Q1

'''
Item 4: Construir la clase TD3. Implementa el entrenamiento del modelo
'''
class TD3(object):
    """
      Clase que implementa el algoritmo de Twin Delayed DDPG (TD3) para el aprendizaje por refuerzo.

      Args:
          state_dim (int):          Dimensión del espacio de estados.
          action_dim (int):         Dimensión del espacio de acciones.
          max_action_value (float): Valor máximo permitido para las acciones.
    """

    def __init__(self, state_dim, action_dim, max_action_value, max_timesteps, initial_lr = 1e-2, priority_memory = False):
        """
          Inicializa la clase TD3.

          Args:
              state_dim (int): Dimensión del espacio de estados.
              action_dim (int): Dimensión del espacio de acciones.
              max_action_value (float): Valor máximo permitido para las acciones.
        """
        self.initial_lr       = initial_lr
        # Definir la tasa de aprendizaje inicial y la función de reducción de la tasa de aprendizaje
        self.lr_lambda = lambda steps: 1 - steps / max_timesteps  # Reducción lineal de la tasa de aprendizaje

        self.actor_net        = Actor (state_dim, action_dim, max_action_value).to (device)
        self.actor_target     = Actor (state_dim, action_dim, max_action_value).to (device)
        self.actor_target.load_state_dict (self.actor_net.state_dict ())
        self.actor_optimizer  = torch.optim.Adam (self.actor_net.parameters (), lr = self.initial_lr) # modificamos tasa aprendizaje ya que vemos que el entrenamiento se estanca
        self.scheduler_actor  = LambdaLR (self.actor_optimizer, self.lr_lambda) # Crear un scheduler para ajustar dinámicamente la tasa de aprendizaje del actor

        self.critic_net       = Critic (state_dim, action_dim).to (device)
        self.critic_target    = Critic (state_dim, action_dim).to (device)
        self.critic_target.load_state_dict (self.critic_net.state_dict ())
        self.critic_optimizer = torch.optim.Adam (self.critic_net.parameters (), lr = self.initial_lr) # modificamos tasa aprendizaje ya que vemos que el entrenamiento se estanca
        self.scheduler_critic  = LambdaLR(self.critic_optimizer, self.lr_lambda) # Crear un scheduler para ajustar dinámicamente la tasa de aprendizaje del actor
        self.max_action       = max_action_value

    def select_action(self, state):
        """
          Método para seleccionar una acción basada en el estado actual.

          Args:
              state (array): Estado actual.

          Returns:
              accion (array): Acción seleccionada.
        """
        state = torch.Tensor (state.reshape(1, -1)).to (device)
        return self.actor_net (state).cpu ().data.numpy ().flatten ()
    #He modificado de 100 a 256 el batch_size = 256, para probar
    def train (self, replay_buffer, iterations, batch_size = 100, discount = 0.99, tau = 0.005,
                                                policy_noise = 0.2, noise_clipping = 0.5, policy_freq = 2):
        """
        Método para entrenar el modelo TD3.

        Args:
            replay_buffer (ReplayBuffer)  : Buffer de repetición que almacena las transiciones de experiencia.
            iterations (int)              : Número de iteraciones de entrenamiento.
            batch_size (int)              : Tamaño del lote de muestras para el entrenamiento.
            discount (float)              : Factor de descuento gamma para la actualización de los valores Q.
            tau (float)                   : Ratio de actualización para el suavizado de parámetros del target.
            policy_noise (float)          : Desviación estándar del ruido gaussiano para la exploración.
            noise_clipping (float)        : Valor máximo del ruido gaussiano añadido a las acciones.
            policy_freq (int)             : Frecuencia de actualización de la política.
        """
        rewards           = []
        losses_critic     = []
        losses_actor      = []
        target_qs         = []  # Almacena los valores objetivo del crítico 1
        target_qs_critic1 = []  # Almacena los valores objetivo del crítico 1
        target_qs_critic2 = []  # Almacena los valores objetivo del crítico 2
        #exploration_factor_values = [] # Registra el valor del factor de exploración en cada iteración
        for it in range (iterations):
            # Tomamos una muestra de transiciones (s, s’, a, r) de la memoria.
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample (batch_size)

            # Pasamos a tensores y lo subimos al 'device' seleccionad, por si podemos acelerar las operaciones con dichos tensores con la gpu
            state      = torch.Tensor(batch_states).to (device)
            next_state = torch.Tensor(batch_next_states).to (device)
            action     = torch.Tensor(batch_actions).to (device)
            reward     = torch.Tensor(batch_rewards).to (device)
            done       = torch.Tensor(batch_dones).to (device)


            #"Target Policy Smoothing" para mejorar la estabilidad del entrenamiento. Esta técnica
            # implica agregar ruido a las acciones seleccionadas por la política objetivo y luego
            # suavizarlas. Sin embargo, este proceso de agregar ruido y suavizar las acciones objetivo
            # no debe contribuir al cálculo del gradiente durante la retropropagación, ya que las acciones
            # objetivo ruidosas son solo para la estimación de los valores Q objetivo y no deben afectar
            #directamente la actualización de los parámetros de la red.
            with torch.no_grad():
                # A partir del estado siguiente s', el Actor del Target ejecuta la siguiente acción a'.
                next_action = self.actor_target (next_state)

                # Añadimos ruido gaussiano a la siguiente acción a' y lo cortamos para tenerlo en el rango de valores aceptado por el entorno.
                # En cada iteración tendremos un ruido diferente. Esto puede ser útil para evitar que el agente se estanque en mínimos locales subóptimos y para descubrir nuevas estrategias efectivas.
                noise = torch.Tensor (batch_actions).data.normal_ (0, policy_noise).to (device)
                noise = noise.clamp (-noise_clipping, noise_clipping)

                next_action = (next_action + noise).clamp (-self.max_action, self.max_action)

                # Los dos Críticos del Target toman un par (s’, a’) como entrada y devuelven dos Q-values Qt1(s’,a’) y Qt2(s’,a’) como salida.
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)

                # Nos quedamos con el mínimo de los dos Q-values: min(Qt1, Qt2). Representa el valor aproximado del estado siguiente.
                target_Q = torch.min (target_Q1, target_Q2)

                # Ecuación de Bellman: Obtenemos el valor objetivo final de los dos Críticos del Modelo, que es: Qt = r + γ * min(Qt1, Qt2), donde γ es el factor de descuento.
                target_Q = reward + ((1 - done) * discount * target_Q).detach()

                target_qs_critic1.append(target_Q1.mean().item())
                target_qs_critic2.append(target_Q2.mean().item())
                target_qs.append(target_Q.mean().item())

            # Los dos Críticos del Modelo toman un par (s, a) como entrada y devuelven dos Q-values Q1(s,a) y Q2(s,a) como salida.
            current_Q1, current_Q2 = self.critic_net (state, action)

            # Calculamos la pérdida procedente de los Crítico del Modelo: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss (current_Q1, target_Q) + F.mse_loss (current_Q2, target_Q)

            # Propagamos hacia atrás la pérdida del crítico y actualizamos los parámetros de los dos Crítico del Modelo con un SGD.
            self.critic_optimizer.zero_grad ()
            critic_loss.backward ()
            self.critic_optimizer.step ()

            # Cada dos iteraciones, actualizamos nuestro modelo de Actor ejecutando el gradiente ascendente en la salida del primer modelo crítico.
            if it % policy_freq == 0:
                actor_loss = -self.critic_net.Q1 (state, self.actor_net (state)).mean ()
                self.actor_optimizer.zero_grad ()
                actor_loss.backward ()
                self.actor_optimizer.step ()

                # cada dos iteraciones, actualizamos los pesos del Actor del Target usando el promedio Polyak.
                for param, target_param in zip (self.actor_net.parameters(), self.actor_target.parameters ()):
                    target_param.data.copy_ (tau * param.data + (1 - tau) * target_param.data)


                for param, target_param in zip (self.critic_net.parameters (), self.critic_target.parameters ()):
                    target_param.data.copy_ (tau * param.data + (1 - tau) * target_param.data)

            # Si tenemos una memoria de repetición con prioridades modificamos las transiciones 
            # if (priority_memory):
                # replay_buffer.add_transitions_to_replay_buffer(critic_loss.cpu(), state.cpu().numpy(), next_state.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), done.cpu().numpy(), batch_size)
            # Almacenamos la recompensa de esta iteración
            rewards.append (reward.mean().item())
            losses_critic.append (critic_loss.mean().item())
            losses_actor.append (actor_loss.mean().item())
            
        # Actualizar el scheduler en cada iteración
        self.scheduler_critic.step ()
        self.scheduler_actor.step ()
        return rewards, losses_critic, losses_actor, target_qs_critic1, target_qs_critic2, target_qs
    # Para almacenar el entrenamiento en disco
    def save (self, filename, directory):
      """
        Guarda el modelo entrenado.

        Argumentos:
            filename (str): Nombre del archivo para guardar el modelo.
            directory (str): Directorio donde se guardará el modelo.
      """
      torch.save (self.actor_net.state_dict (), "%s/%s_actor.pth" % (directory, filename))
      torch.save (self.critic_net.state_dict (), "%s/%s_critic.pth" % (directory, filename))

    # Cargamos desde disco el entrenamiento realizado del modelo
    def load (self, filename, directory):
      """
        Carga un model  o entrenado.

        Argumentos:
            filename (str): Nombre del archivo que contiene el modelo guardado.
            directory (str): Directorio donde se encuentra el modelo guardado.
      """
      self.actor_net.load_state_dict (torch.load("%s/%s_actor.pth" % (directory, filename)))
      self.critic_net.load_state_dict (torch.load("%s/%s_critic.pth" % (directory, filename)))
'''
Item 5. Creamos una función para poder evaluar la política medainte el promedio de las recompenas objtenidas en cada episodio. Nota: Cada 10 episodios realiamos una evaluación. También nos creamos las funciones auxiliares para serializar y deserializar infornación.
'''

###############
# Nos creamos la función que nos evalua las recompensas de 10 epocas (configurable)
##############
def evaluate_train_policy (policy, env, eval_episodes = 10):
  """
      Función para evaluar el desempeño de una política en un entorno de RL.

      Args:
          policy: Objeto que representa la política a evaluar.
          eval_episodes: Número de episodios de evaluación a ejecutar (por defecto, 10).

      Returns:
          avg_reward (float): Recompensa promedio obtenida durante los episodios de evaluación.
  """
  avg_reward = 0.
  reward_p = []
  for _ in range(eval_episodes):
    obs  = env.reset()
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
      reward_p.append (reward)

  avg_reward /= eval_episodes
  print ("-------------------------------------------------")
  print ("Recompensa promedio en el paso de Evaluación: %f" % (avg_reward))
  print ("-------------------------------------------------")

  return avg_reward

def evaluate_policy (policy, env, eval_episodes = 10):
  """
      Función para evaluar el desempeño de una política en un entorno de RL.

      Args:
          policy: Objeto que representa la política a evaluar.
          eval_episodes: Número de episodios de evaluación a ejecutar (por defecto, 10).

      Returns:
          avg_reward (float): Recompensa promedio obtenida durante los episodios de evaluación.
  """
  avg_reward = 0.
  reward_p = []
  for _ in range(eval_episodes):
    obs  = env.reset()
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
      reward_p.append (reward)

  avg_reward /= eval_episodes
  print ("-------------------------------------------------")
  print ("Recompensa promedio en el paso de Evaluación: %f" % (avg_reward))
  print ("-------------------------------------------------")

  return reward_p, range (eval_episodes)
def serialize_object (obj, file_name):
    """
    Serialize an object using pickle and save it to a file.

    Parameters:
    - obj: The object to be serialized.
    - file_name: The name of the file to save the serialized object.
    """

    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)
    print(f"Object serialized and saved to '{file_name}'.")

def deserialize_object(file_name):
    """
    Deserialize an object from a file using pickle.

    Parameters:
    - file_name: The name of the file containing the serialized object.

    Returns:
    - The deserialized object.
    """
    with open(file_name, 'rb') as file:
        deserialized_object = pickle.load(file)

    return deserialized_object

def lists_to_serializable_object(lists, attribute_names, file_name):
    """
    Convert a set of lists into a serializable object and save it to a file.

    Parameters:
    - lists: A list of lists to be converted into a serializable object.
    - attribute_names: A list of attribute names for the lists.
    - file_name: The name of the file to save the serialized object.
    """
    serializable_object = {}
    for lst, attribute_name in zip(lists, attribute_names):
        serializable_object[attribute_name] = lst
    with open(file_name, 'wb') as file:
        pickle.dump(serializable_object, file)
    print(f"Set of lists converted into a serializable object and saved to '{file_name}'.")
# Guardamos el tiempo que ardo el entrenamiento
def serialize_training (tiempo_inicio, tiempo_fin, total_steps,version):
    """
    Función para serializar los datos de entrenamiento con el tiempo de entrenamiento
    y el número total de pasos.

    Args:
    - tiempo_inicio: Tiempo inicial de referencia para medir el tiempo de entrenamiento.
    - tiempo_fin : Tiempo inicfinal de referencia para medir el tiempo de entrenamiento.
    - total_steps: Número total de pasos de entrenamiento.

    Returns:
    - Nada. Simplemente guarda los datos serializados en un archivo.
    """


    # Calcular la duración total del entrenamiento en segundos y minutos
    tiempo_entrenamiento_segundos = tiempo_fin - tiempo_inicio
    tiempo_entrenamiento_minutos = tiempo_entrenamiento_segundos / 60

    # Crear un diccionario con la información que deseamos serializar
    datos_entrenamiento = {
        "tiempo_entrenamiento_segundos": tiempo_entrenamiento_segundos,
        "tiempo_entrenamiento_minutos": tiempo_entrenamiento_minutos,
        "numero_pasos": total_steps,
        # Otros datos que desees serializar...
    }

    # Serializar los datos en un archivo utilizando pickle
    nombre_archivo = f'./results/datos_entrenamiento_{version}.pkl'
    with open(nombre_archivo, 'wb') as f:
        pickle.dump(datos_entrenamiento, f)

    print(f"Datos de entrenamiento serializados correctamente en '{nombre_archivo}'.")
'''
 Item 6: Creamos las carpetas y nombres de fichero donde se almacenarán los modelos y su entrenamiento                                                 
'''
def created_models_directory (env_name,seed, save_models,version):
    file_model_name = "%s_%s_%s_%s" % ("TD3", env_name, str(seed),version)
    print ("---------------------------------------")
    print ("Fichero de los modelos entrenados: %s" % (file_model_name))
    print ("---------------------------------------")
    
    # Resultados del entrenamiento
    if not os.path.exists("./results"):
      os.makedirs("./results")
    
    # Modelos creados
    if save_models and not os.path.exists("./pytorch_models"):
      os.makedirs("./pytorch_models")
    return file_model_name
'''
Item 7: Cargamos el entorno deseado y obtenemos información de sus estados y acciones
'''
# Directorio donde se almacenarán los resultados finales del entrenamiento
def mkdir(base, name):
    """
      Crea un nuevo directorio si no existe en la ubicación especificada.

      Args:
          base (str): Ruta base donde se creará el nuevo directorio.
          name (str): Nombre del nuevo directorio a crear.

      Returns:
          str: Ruta completa del nuevo directorio creado.
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


###############
# Cargamos el entorno
###############


class noisy_action_wrapper (gym.ActionWrapper):
    """
    Envoltorio (wrapper) que agrega ruido gaussiano a las acciones seleccionadas antes de ejecutarlas en el entorno.

    Args:
        env (gym.Env): Entorno original al que se aplicará el envoltorio.
        noise_level (float): Nivel de ruido gaussiano a agregar a las acciones (por defecto: 0.1).
    """
    def __init__ (self, env, noise_level = 0.1):
        """
        Inicializa el envoltorio.

        Args:
            env (gym.Env): Entorno original al que se aplicará el envoltorio.
            noise_level (float): Nivel de ruido gaussiano a agregar a las acciones (por defecto: 0.1).
        """
        super (noisy_action_wrapper, self).__init__ (env)
        self.noise_level = noise_level

    def action (self, action):
        """
        Agrega ruido gaussiano a las acciones seleccionadas.

        Args:
            action (np.ndarray): Acción seleccionada por la política.

        Returns:
            np.ndarray: Acción con ruido agregado.
        """
        # Agregar ruido gaussiano a las acciones
        noisy_action = action + np.random.normal (0, self.noise_level, size = action.shape)

        # Recortar las acciones para asegurarse de que estén dentro de los límites del espacio de acción del entorno
        return np.clip (noisy_action, self.action_space.low, self.action_space.high)


# Guardamos el entorno de entrenamiento, para no empezar desde cero siempre
def save_env (filename, directory, env):
    """
    Guarda el modelo entrenado y el estado del entorno.

    Argumentos:
        filename (str): Nombre del archivo para guardar el modelo.
        directory (str): Directorio donde se guardará el modelo.
        env: Objeto del entorno.
    """
    # Guardar el estado del entorno
    with open(f"{directory}/{filename}_env.pkl", "wb") as f:
        pickle.dump(env, f)

#Cargo el entorno de entrenamiento para empezar donde se quedo...
def load_env(filename, directory, gym = None, env_name = None):
    """
    Carga un modelo entrenado y el estado del entorno si los archivos existen.

    Argumentos:
        env_name (str): Nombre del entorno a cargar.
        filename (str): Nombre del archivo que contiene el modelo guardado.
        directory (str): Directorio donde se encuentra el modelo guardado.

    Returns:
        env: Objeto del entorno si los archivos existen, None en caso contrario.
    """
    env_path    = f"{directory}/{filename}_env.pkl"

    # Cargar el estado del entorno si existe
    if os.path.exists(env_path):
        with open(env_path, "rb") as f:
            env = pickle.load(f)
        print(f"Entorno '{env_name}' cargado correctamente.")
        return env
    else:
        print(f"No se encontró el archivo del entorno '{env_name}'.")
        if gym is not None and env_name is not None:
          return gym.make (env_name)
        else:
            return None

# Guardamos el tiempo que ardo el entrenamiento
def serializar_entrenamiento (tiempo_inicio, tiempo_fin, total_steps,version):
    """
    Función para serializar los datos de entrenamiento con el tiempo de entrenamiento
    y el número total de pasos.

    Args:
    - tiempo_inicio: Tiempo inicial de referencia para medir el tiempo de entrenamiento.
    - tiempo_fin : Tiempo inicfinal de referencia para medir el tiempo de entrenamiento.
    - total_steps: Número total de pasos de entrenamiento.

    Returns:
    - Nada. Simplemente guarda los datos serializados en un archivo.
    """


    # Calcular la duración total del entrenamiento en segundos y minutos
    tiempo_entrenamiento_segundos = tiempo_fin - tiempo_inicio
    tiempo_entrenamiento_minutos = tiempo_entrenamiento_segundos / 60

    # Crear un diccionario con la información que deseamos serializar
    datos_entrenamiento = {
        "tiempo_entrenamiento_segundos": tiempo_entrenamiento_segundos,
        "tiempo_entrenamiento_minutos": tiempo_entrenamiento_minutos,
        "numero_pasos": total_steps,
        # Otros datos que desees serializar...
    }

    # Serializar los datos en un archivo utilizando pickle
    nombre_archivo = f'./results/datos_entrenamiento_{version}.pkl'
    with open(nombre_archivo, 'wb') as f:
        pickle.dump(datos_entrenamiento, f)

    print(f"Datos de entrenamiento serializados correctamente en '{nombre_archivo}'.")

def plot_training_metrics (evaluations, avg_rewards,  avg_losses_c,avg_losses_a, all_losses_c,
                           all_losses_a, target_qs_c1,
                           target_qs_c2, target_qs, n_steps_epochs, version, epochs, steps):
    """
      Plotea las métricas de entrenamiento y guarda el gráfico en disco.

      Args:
          evaluations:_  Evaluación de la politica creada hasta el momento
          avg_rewards (list): Lista de recompensas promedio.
          avg_losses_c (list): Lista de pérdidas promedio para el crítico.
          avg_losses_a (list): Lista de pérdidas promedio para el actor.
          all_losses_c (list): Lista de todas las pérdidas para el crítico.
          all_losses_a (list): Lista de todas las pérdidas para el actor.
          target_qs_c1 (list): Lista de valores objetivo para el crítico 1.
          target_qs_c2 (list): Lista de valores objetivo para el crítico 2.
          target_qs (list): Lista de valores objetivo.
          n_steps_epochs (int): Número de pasos por época.
          version (str): Versión del modelo o del algoritmo.
          epochs (int): Número total de épocas.
          steps (int): Número total de pasos.

      Returns:
          None
    """

     # Graficar evaluations
    plt.figure(figsize=(8, 6))
    plt.plot(evaluations)
    plt.title('Evaluación del a porlítica creada')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa De al politica')
    plt.grid(True)
    plt.savefig(f'./results/evaluations_plot__{epochs}_{steps}_{version}.png')
    plt.close()

    # Graficar la recompensa promedio
    plt.figure(figsize=(8, 6))
    plt.plot(avg_rewards)
    plt.title('Recompensa Promedio por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Promedio')
    plt.grid(True)
    plt.savefig(f'./results/avg_rewards_plot__{epochs}_{steps}_{version}.png')
    plt.close()

    # Graficar la pérdida promedio del crítico
    plt.figure(figsize=(8, 6))
    plt.plot(avg_losses_c)
    plt.title('Pérdida Promedio del Crítico por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Pérdida Promedio del Crítico')
    plt.grid(True)
    plt.savefig(f'./results/avg_losses_c_plot__{epochs}_{steps}_{version}.png')
    plt.close()

    # Graficar la pérdida promedio del actor
    plt.figure(figsize=(8, 6))
    plt.plot(avg_losses_a)
    plt.title('Pérdida Promedio del Actor por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Pérdida Promedio del Actor')
    plt.grid(True)
    plt.savefig(f'./results/avg_losses_a_plot__{epochs}_{steps}_{version}.png')
    plt.close()

    # Graficar la pérdida del crítico
    plt.figure(figsize=(8, 6))
    plt.plot(all_losses_c)
    plt.title('Pérdida Promedio del Crítico por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Pérdida Promedio del Crítico')
    plt.grid(True)
    plt.savefig(f'./results/all_losses_c_plot__{epochs}_{steps}_{version}.png')
    plt.close()

    # Graficar la pérdida promedio del actor
    plt.figure(figsize=(8, 6))
    plt.plot(all_losses_a)
    plt.title('Pérdida Promedio del Actor por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Pérdida Promedio del Actor')
    plt.grid(True)
    plt.savefig(f'./results/all_losses_a_plot__{epochs}_{steps}_{version}.png')
    plt.close()



    # Graficar valores objetivo del C1
    plt.figure(figsize=(8, 6))
    plt.plot(target_qs_c1)
    plt.title('valores objetivo del C1 por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('valores objetivo del C1')
    plt.grid(True)
    plt.savefig(f'./results/target_qs_c1_plot__{epochs}_{steps}_{version}.png')
    plt.close()

    # Graficar objetivo del C2
    plt.figure(figsize=(8, 6))
    plt.plot(target_qs_c2)
    plt.title('Valores objetivo del C2 por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Valores objetivo del C2')
    plt.grid(True)
    plt.savefig(f'./results/target_qs_c2_plot__{epochs}_{steps}_{version}.png')
    plt.close()

    # Graficar valores Objetivo
    plt.figure(figsize=(8, 6))
    plt.plot(target_qs)
    plt.title('Valores Objetivo por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('valores Objetivo')
    plt.grid(True)
    plt.savefig(f'./results/target_qs_plot__{epochs}_{steps}_{version}.png')
    plt.close()

    # Graficar n_steps_epochs
    plt.figure(figsize=(8, 6))
    plt.plot(n_steps_epochs)
    plt.title('Pasos por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Pasos')
    plt.grid(True)
    plt.savefig(f'./results/n_steps_epochs_plot__{epochs}_{steps}_{version}.png')
    plt.close()

#############################
## aplanamos la estrutura
#############################
def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def create__metrics_imagen (evaluations, avg_rewards,  avg_losses_c,avg_losses_a, all_losses_c,
                                                    all_losses_a, target_qs_c1,
                                                    target_qs_c2, target_qs,
                                                    n_steps_epochs, version, epochs, steps):

  steps_to_plot = {100,200, 100000, 250000, 500000, 1000000, 1250000, 1500000, 2000000, 2250000, 2500000, 3000000, 3250000, 3500000, 4000000, 4250000, 4500000, 5000000}
  if steps in steps_to_plot:
    # Llama a la función plot_training_metrics para crear y guardar los gráficos
    plot_training_metrics (evaluations, avg_rewards, avg_losses_c, avg_losses_a, np.concatenate(all_losses_c),
                          np.concatenate(all_losses_a), np.concatenate(target_qs_c1), np.concatenate(target_qs_c2),flatten_list (target_qs),
                          n_steps_epochs,version,  epochs, steps)   

'''
                                                  
'''