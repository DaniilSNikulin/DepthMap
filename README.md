# DepthMap

## Мотивация

### Карта (матрица) глубины
Матрица глубины (depth map) - это некоторая матрица, каждый элемент которой содержит дальность до объекта.
Другими словами, если имеется некотороые изображение, то матрица глубины говорит о том, какого расстояние до каждого нарисованного объекта.
С помощью матрицы глубины воссоздается трехмерная модель пространства.

### Проблема

Создание 3D модели пространства невозможно без карты глубины.
Самое простое и распространенное решение - это использовать стереозрение или радар.
Для стереозрения нужна очень точная калибровка, которая может легко нарушиться со временем.
Радары очень хороши, но они дорогие.

Дополнительно оба даных подхода не применимы для создание 3D видео из уже отснятых 2D фильмов,
что в последнее время стало очень популярно.

### Задача

Написать скрипт, который строит матрицу глубины для набора кубиков с одного изображения, без использования бинокулярного зрения или радаров.



## Состав команды:
* Никулин Даниил
* Гостевский Дмитрий
* Лапко Данила

## Запуск
```
python main.py -i input_file.png -o output_file.png
```
Формат не так важен, скрипт работает со всеми форматами, которые поддерживает OpenCV.

Если хочется только посмотреть результат, то тогда имя выходного файла и опцию ``-o`` можно опустить. Тогда результирующее изображение будет показано на экране.

Есть дополнительный ключ: ``-s``. Если его указать, то будет показана визуализация некоторых промежуточныъ этапов обработки.



## Зависимости
Для скрипта необходимы следующие Python библиотеки:
* matpotlib
* cv2
* numpy
* math
* getopt

### Установка OpenCV
```
conda install -c menpo opencv
```



## Этапы работы скрипта
* Нахождение всех кубиков на картинке
* выделение каждого кубика в отдельное изображение
* edge detection
* Hough transform (нахождение линий)
* выделение псевдо параллельных прямых
* нахождение точек схода
* нахождение ближайшей точки к зрителю
* построение функции градиента глубины
* совмещение нескольких картинок


## Список известных багов:
* Высокая чувствительность к шумам и постороним предметам
* Градиент глубины считается очень долго
* Яркие тени детектируются как объект
* В случае обработки нескольких кубиков подробный режим показывает этапы только для самого большого кубика
* Не работает в случае, когда кубик частично закрыт другим кубиком
* Все предметы должны находиться на одной плоскости, а сама плоскость должна быть горизонтальна

[Ссылка на видео и презентацию](https://yadi.sk/d/Kv9pmqPk3PRcgA)

