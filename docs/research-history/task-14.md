# История TASK-14 до сокращения карточки

Не читать при обычном старте задачи. Сохранено без сокращения 17 октября2026.
Актуальное состояние находится в требованиях и короткой карточке.

## Прежний план



## Прежние заметки

TASK10 now includes five-seed paired heat3d hard/soft validation. This is partial evidence only:all-PDE5seed composition/leave-one-out and Adam->LBFGS dependency remain pending. Do not mark AC1/AC3 or change defaults.

Adam->LBFGS dependency complete. Preliminary hard+LB100 combination2seeds:heat3d global0.086/0.109%,final0.317/0.330%; matchedwall~43s controlsfinal0.952/0.761%. Heat2d/Burgers combinations improve; advectionsoft+LB tested. Memoryhard+49%,no defaultpromotion. Five-seedallPDE andleave-one-out/fullbatchAdam ablation pending.
