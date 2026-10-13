# PINN runner

Одноразовый файловый решатель задач на основе Physics-Informed Neural Networks.
Вход: `config.yaml`, `functions.py` и checkpoint для предсказания/дообучения.
Runner создаёт сеть, обучает её по PDE/начальным/граничным условиям и сохраняет
результат. Эталонный `data.mat` не требуется.

Поддерживаемый путь: непрерывное время, прямоугольные области 1D/2D/3D,
несколько скалярных входных параметров и явные условия Дирихле/Неймана/Робина
или теплового потока. Предсказание допускает конкретные значения параметров. Коэффициенты можно
задавать формулами и фиксированными пространственными таблицами NPZ.

- [Полный API: YAML, Problem API, запуск и результаты](docs/API.md)
- [Пространственные коэффициенты: формулы и таблицы](docs/coefficients.md)
- [Архитектура и расширение](docs/architecture.md)
- [Проверки разработки](docs/development.md)
- [План развития и критерии готовности](docs/roadmap.md)
- [Windows и CUDA: подготовка и проверки](docs/cuda-testing.md)
- [Аналитические PDE-бенчмарки и регрессии обучения](docs/numerical-benchmarks.md)
- [Фоновые GPU-задания: start/status/collect](docs/gpu-jobs.md)
- [CPU-тесты](tests/README.md)
- [2D теплопроводность](examples/heat2d)
- [2D теплопроводность с табличным полем](examples/heat_spatial_coefficient)
- [Семейство задач с варьирующейся теплопроводностью](examples/heat_parameterized)

```sh
python run.py --mode train --input-dir examples/heat_parameterized --output-dir /tmp/pinn-result
python run.py --mode predict --input-dir examples/heat_parameterized --output-dir /tmp/pinn-prediction --checkpoint /tmp/pinn-result/checkpoints/final.ckpt
```

Установите зависимости по development.md. Примеры используют CPU и только
три шага, чтобы быстро проверить запуск. Production-обучение планируется на
CUDA, а разработка и проверки выполняются на CPU.

Проект развивается из [pinns-torch](https://github.com/rezaakb/pinns-torch).
Сведения о лицензии сохранены в репозитории.
