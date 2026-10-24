
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

IMG_SIZE = (64, 64)        # Размер изображений в датасете
BATCH_SIZE = 32            # Размер батча
EPOCHS = 15                # Количество эпох обучения
NUM_CLASSES = 7            # Количество классов комплектующих

print("-" * 60)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Классы:", list(train_generator.class_indices.keys()))
print(f"Обучающая выборка:  {train_generator.samples} изображений")
print(f"Валидационная:      {val_generator.samples} изображений")
print(f"Тестовая:           {test_generator.samples} изображений")


#Определение 5ти моделей

def create_model_1_mlp():
#Модель 1: Полносвязная нейронная сеть (без свёрточных слоёв).
    return models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name='MLP')


def create_model_2_cnn_basic():
#Модель 2: Базовая свёрточная сеть (2 свёрточных блока).
    return models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name='CNN_basic')


def create_model_3_cnn_deep():
#Модель 3: Глубокая свёрточная сеть (3 свёрточных блока + Dropout)
    return models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name='CNN_deep')


def create_model_4_cnn_aug():
#Модель 4: Та же глубокая сеть, но обучается на аугментированных данных.
    return create_model_3_cnn_deep()


def create_model_5_transfer():
#Модель 5: Перенос обучения на основе MobileNetV2.
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(64, 64, 3)
    )
    base.trainable = False
    return models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name='MobileNetV2')

model_creators = [
    ('МЛП (без свёрток)',              create_model_1_mlp),
    ('СНС (2 блока)',                  create_model_2_cnn_basic),
    ('СНС (3 блока + Dropout)',        create_model_3_cnn_deep),
    ('СНС (3 блока) + Аугментация',    create_model_4_cnn_aug),
    ('Перенос обучения (MobileNetV2)', create_model_5_transfer)
]

aug_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)
train_gen_aug = aug_datagen.flow_from_directory(
    'data/train', target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='training'
)
val_gen_aug = aug_datagen.flow_from_directory(
    'data/train', target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='validation'
)


#Обучение всех моделей
histories = []
models_list = []

for i, (name, creator) in enumerate(model_creators, start=1):
    print("\n" + "-" * 60)
    print(f"Модель {i}/5: {name}")
    print("-" * 60)

    model = creator()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    if 'Аугментация' in name:
        tr, vl = train_gen_aug, val_gen_aug
    else:
        tr, vl = train_generator, val_generator

    history = model.fit(
        tr,
        epochs=EPOCHS,
        validation_data=vl,
        verbose=1
    )
    histories.append(history)
    models_list.append(model)


print("\n" + "-" * 80)
print("Оценка моделей на тестовой выборке")
print("-" * 80)

results = []
for i, (name, _) in enumerate(model_creators):
    model = models_list[i]
    history = histories[i]

    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]

    results.append({
        'Модель': name,
        'Точность (обучение)': round(train_acc, 4),
        'Точность (валидация)': round(val_acc, 4),
        'Точность (тест)': round(test_acc, 4),
        'Переобучение': round(train_acc - test_acc, 4),
        'Параметров': model.count_params()
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))

df.to_csv('results.csv', index=False, encoding='utf-8-sig')
print("\nТаблица сохранена в файл results.csv")


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, history in enumerate(histories):
    ax = axes[i]
    ax.plot(history.history['accuracy'], label='Обучение', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Валидация', linewidth=2)
    ax.set_title(model_creators[i][0], fontsize=11)
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Точность')
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle('Сравнение 5 архитектур нейронных сетей\nна датасете комплектующих ПК',
             fontsize=14, fontweight='bold', y=0.995)

axes[-1].remove()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("График сохранён в файл model_comparison.png")
plt.show()


best_idx = int(np.argmax([r['Точность (тест)'] for r in results]))
best_model = models_list[best_idx]
best_name = model_creators[best_idx][0]

print("\n" + "-" * 60)
print(f"Лучшая модель: {best_name}")
print("-" * 60)

# Предсказания для тестовой выборки
test_generator.reset()
y_pred_proba = best_model.predict(test_generator, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Матрица ошибок — {best_name}', fontsize=13)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Матрица ошибок сохранена в файл confusion_matrix.png")
plt.show()

print("\nОтчёт по классификации для лучшей модели:")
print(classification_report(y_true, y_pred,
                            target_names=class_labels,
                            digits=3))
print("-" * 60)
print("Получены файлы:")
print("  - results.csv           — таблица сравнения моделей")
print("  - model_comparison.png  — графики обучения 5 моделей")
print("  - confusion_matrix.png  — матрица ошибок лучшей модели")