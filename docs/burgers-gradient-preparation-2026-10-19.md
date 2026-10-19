# Бюргерс: одинаковая невязка ещё не означает правильный градиент

Это предварительный контроль существующего оператора для будущей TASK-34,
не приёмка качества. Обучения, новой сети, CUDA и переноса ускорений heat нет.
TASK-34 остаётся To Do,0/4. Использован существующий
`benchmarks/problems/burgers/functions.py`, вязкость nu=0,1.

## Постановка и независимый скалярный пробник

В единичном пространственном интервале точное решение и источник заданы:

$$
u_*=e^{-t}\sin(\pi x),\qquad
u_t+u u_x-\nu u_{xx}=f,\qquad
f=e^{-t}(-1+\nu\pi^2)\sin(\pi x)
+\pi e^{-2t}\sin(\pi x)\cos(\pi x).
$$

Начальное условие — sin(pi·x), на обоих концах нулевое значение.
Оно совместимо с IC, включая углы. Для любого a скалярный пробник сохраняет
эти данные:

$$
u_a=h(t)\sin(\pi x),\qquad h=e^{-t}+(a-1)t.
$$

Обозначим S=sin(pi·x), C=cos(pi·x), delta=a−1. Независимое раскрытие даёт:

$$
R_a=\delta(1+\nu\pi^2t)S+\pi(h^2-e^{-2t})SC,
\qquad
\partial_aR_a=(1+\nu\pi^2t)S+2\pi htSC.
$$

Если первый множитель в u·u_x ошибочно отсоединить от графа, значения
поля/невязки не изменятся, но из производной пропадёт pi·h·t·S·C.
Для L=mean(R²) верная производная равна2·mean(R·partial_a R).
На точном решении R=0 обе проверки градиента вырождаются; поэтому нужен
ненулевой пробник a0,7, а не только a1.

## Полученный результат CPU FP64

20 точек: x0/0,17/0,43/0,83/1 ×t0/0,13/0,31/0,5.
Верный градиент согласуется с формулой при rtol1e-10/atol1e-11.
Неверная ветвь имеет побайтово одинаковую невязку, но при a0,7:

| Производная L по a | Значение |
|---|---:|
| Правильная и независимая формула |−0,3921648720662|
| После ошибочного отсоединения |−0,3487378874459|
| Абсолютная разница |0,0434269846203|

При a1 максимум невязки1,11e-16, обе производные практически нулевые —
только такой контроль ошибку не обнаружил бы. Это скалярный диагностический
параметр, не все веса будущей сети и не ошибка решения обученной PINN.
[Машинный срез и SHA](experiments/2026-10-19-burgers-gradient-probe.json).

## Воспроизведение из корня в проектном Python

```python
import math
import torch
from benchmarks.cases import CASES
from pinnstorch.api import Coordinates, Equation
from pinnstorch.runner.problem import load_problem

problem = load_problem(CASES["burgers"].source)
xx, tt = torch.meshgrid(
    torch.tensor([0., .17, .43, .83, 1.], dtype=torch.float64),
    torch.tensor([0., .13, .31, .5], dtype=torch.float64), indexing="ij")
rows = []
for a in (1., .7):
    x, t = xx.reshape(-1, 1).clone().requires_grad_(), tt.reshape(-1, 1).clone().requires_grad_()
    theta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    h, s, c = (-t).exp() + (theta-1)*t, (math.pi*x).sin(), (math.pi*x).cos()
    u = h*s
    e = Equation({"u": u}, Coordinates({"x": x}, t))
    r = problem.equations["burgers"](e)
    wrong = r + (u.detach()-u)*e.diff("u", "x")
    assert torch.equal(wrong, r)
    closed_r = (theta-1)*(1+.1*math.pi**2*t)*s + math.pi*(h*h-(-2*t).exp())*s*c
    torch.testing.assert_close(r, closed_r, rtol=1e-10, atol=1e-11)
    gradient = torch.autograd.grad(r.square().mean(), theta, retain_graph=True)[0]
    wrong_gradient = torch.autograd.grad(wrong.square().mean(), theta, retain_graph=True)[0]
    expected = 2*(r*((1+.1*math.pi**2*t)*s + 2*math.pi*h*t*s*c)).mean()
    torch.testing.assert_close(gradient, expected, rtol=1e-10, atol=1e-11)
    if a == .7:
        assert abs(float(gradient-wrong_gradient)) > 1e-4
    rows.append(dict(amplitude=a, residual_max=float(r.detach().abs().max()),
                     gradient=float(gradient), detached_gradient=float(wrong_gradient)))
print(rows)
```

Этот контроль следует включить в будущую проверку нелинейной невязки и
весовых градиентов. Для настоящего результата TASK-34 ещё нужны независимые
пространственно-временные ошибки, корректная входная граница переноса,
заранее заданная CUDA-матрица нескольких состояний и файловые проверки.
Текущие тесты теплопроводности этого не заменяют.
