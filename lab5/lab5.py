import numpy as np
import random
import sys
from keras import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM, Activation
from keras._tf_keras.keras.optimizers import RMSprop
from keras._tf_keras.keras.layers  import Input


# Загрузка и подготовка текста
text_list = []
with open("/Users/admin/PycharmProjects/reinforcement_learning/diploma.txt", 'r', encoding='utf8') as sh_file:
    for line in sh_file:
        if line != '\n':
            text_list.append(line)

text = ''.join([line.lower() for line in text_list])

# Создаем словарь символов
chars = sorted(list(set(text)))
char_indices = {c: i for i, c in enumerate(chars)}
indices_char = {i: c for i, c in enumerate(chars)}

# Параметры для модели
maxlen = 40
step = 3

# Генерация обучающих последовательностей
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])

# Преобразование в one-hot вектора
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)  # or use np.bool_
y = np.zeros((len(sentences), len(chars)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Создание модели LSTM
model = Sequential()
model.add(Input(shape=(maxlen, len(chars))))  # Add Input layer as the first layer
model.add(LSTM(units=256))
model.add(Dense(units=len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Сохраняем структуру модели
model_structure = model.to_json()
with open("/Users/admin/PycharmProjects/ml/reinforcement_learning/diss_lstm_model.json", 'w') as json_file:
    json_file.write(model_structure)

# Обучение модели
epochs = 10
batch_size = 128

for i in range(5):
    model.fit(X, y, batch_size=batch_size, epochs=epochs)
    model.save_weights(f'/Users/admin/PycharmProjects/ml/reinforcement_learning/diss_lstm_weights_{i + 1}.weights.h5')

# Функция для генерации текста
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def text_gen(diversities=[.2]):
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in diversities:
        print('\n----- diversity:', diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

# Генерация текста с разными значениями "diversity"
text_gen(diversities=[.2, .5, 1.0, 1.2])