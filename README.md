# DepthMap

Построение матрицы глубины для двумерных изображений.
	
	
Состав команды:
* Никулин Даниил
* Гостевский Дмитрий
* Лапко Данила

Матрица глубины (depth map) - это некоторая матрица, каждый элемент которой содержит дальность до объекта. Другими словами, если имеется некотороые изображение, то матрица глубины говорит о том, какого расстояние до каждого нарисованного объекта.
Хочется строить матрицу глубины с одного изображения, без использования бинокулярного зрения или радаров.


Этапы работы скрипта
* Нахождение всех кубиков на картинке
* выделение каждого кубика в отдельное изображение
* edge detection
* Hough transform (нахождение линий)
* выделение псевдо параллельных прямых
* нахождение точек схода
* нахождение ближайшей точки к зрителю
* построение функции градиента глубины
* совмещение нескольких картинок



Список известных багов: 
* Высокая чувствительность к шумам и постороним предметам
* Градиент глубины считается очень долго
* Яркие тени детектируются как объект

[Ссылка на видео](https://yadi.sk/d/Kv9pmqPk3PRcgA)

