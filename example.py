import numpy as np
import pinocchio as pin
from scipy.integrate import odeint

# 1. Создаем модель (например, 2-звенный манипулятор)
model = pin.Model()
model.gravity.linear = np.array([0, 0, -9.81])

# Добавляем два звена с вращательными суставами
joint1 = pin.JointModelRX()
joint2 = pin.JointModelRY()
inertia1 = pin.Inertia.FromSphere(1.0, 0.5)
inertia2 = pin.Inertia.FromSphere(1.0, 0.5)

model.addJoint(0, joint1, pin.SE3.Identity(), "joint1")
model.appendBodyToJoint(1, inertia1, pin.SE3.Identity())
model.addJoint(1, joint2, pin.SE3.Identity(), "joint2")
model.appendBodyToJoint(2, inertia2, pin.SE3.Identity())

data = model.createData()

# 2. Параметры симуляции
q_current = np.array([0.0, 0.0])  # Текущая конфигурация (начальная поза)
q_target = np.array([1.0, 0.5])   # Целевая конфигурация
dt = 0.01                         # Шаг симуляции
k_spring = 100.0                  # Жесткость пружины
damping = 2.0                     # Демпфирование

# 3. Функция для расчета сил и производных состояния
def dynamics(y, t):
    q, qdot = y[:model.nq], y[model.nq:]
    
    # Прямая кинематика для центров масс
    pin.forwardKinematics(model, data, q)
    com1 = data.oMi[1].translation  # Центр масс 1-го звена
    com2 = data.oMi[2].translation  # Центр масс 2-го звена
    
    # Целевые позиции центров масс (при q_target)
    pin.forwardKinematics(model, data, q_target)
    com1_target = data.oMi[1].translation
    com2_target = data.oMi[2].translation
    
    # Рассчитываем "пружинные" силы
    force1 = k_spring * (com1_target - com1) - damping * qdot[0] * np.array([1, 0, 0])
    force2 = k_spring * (com2_target - com2) - damping * qdot[1] * np.array([0, 1, 0])
    
    # Создаем объект внешних сил
    fext = pin.StdVec_Force()
    fext.extend([pin.Force.Zero() for _ in range(model.njoints)])
    fext[1] = pin.Force(force1, np.zeros(3))  # Сила для 1-го звена
    fext[2] = pin.Force(force2, np.zeros(3))  # Сила для 2-го звена
    
    # Вычисляем ускорения
    pin.computeAllTerms(model, data, q, qdot)
    ddq = pin.aba(model, data, q, qdot, fext)
    
    return np.concatenate([qdot, ddq])

# 4. Интеграция уравнений движения
t_span = np.arange(0, 2.0, dt)
y0 = np.concatenate([q_current, np.zeros(model.nv)])
result = odeint(dynamics, y0, t_span)

# 5. Визуализация (используем MeshCat)
from meshcat import Visualizer
from pinocchio.visualize import MeshcatVisualizer

viz = MeshcatVisualizer(model, model, model)
viz.initViewer(open=True)
viz.loadViewerModel()

for q in result[:, :model.nq]:
    viz.display(q)
    pin.forwardKinematics(model, data, q)
    print("Current COM positions:", data.oMi[1].translation, data.oMi[2].translation)
