# Данные для тестового задания

Все 18,454 записей сохранены в исходном порядке, ID непустые и уникальные.

- `tracks.csv`: `track_id`, исходные метаданные, `track_popularity`, исходные `lyrics`, а также `tfidf_row`, `lyrics_status`, `has_lyrics`.
- `audio_features.csv`: `track_id` и все 12 исходных аудиопризнаков: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`. Числа сохранены с исходной точностью и шкалой. `key` и `mode` также сохранены. Популярность находится в `tracks.csv`, поскольку это метаданные, а не свойство аудиосигнала.
- `text_tfidf.npz`: SciPy CSR, float32, размер 18,454 × 102,102. Словарь, ID и параметры хранятся внутри того же файла, дополнительных основных файлов нет.

## Соответствие строк и ID

`tfidf_row` — индекс строки матрицы с нуля. Строка матрицы `X[i]` принадлежит записи `tracks.csv` с `tfidf_row == i`. Исходно обе CSV имеют один порядок `track_id`. CSV содержат заголовок, он не считается записью. При сортировке таблицы используйте `tfidf_row`, а для объединения CSV — `track_id`.

В NPZ дополнительно сохранены массивы `track_ids` (порядок строк), `feature_names` (порядок столбцов), `idf` и строка `config_json`. Они не мешают обычному `scipy.sparse.load_npz` и читаются без pickle.

## Отсутствующие тексты

Исходный `lyrics` сохранён дословно. `lyrics_status`: {"available": 18158, "instrumental_marker": 16, "missing_marker": 280}.
`missing_empty` означает пустое поле. `missing_marker` — поле, полностью состоящее из маркеров NA/N/A/NaN/null/none/missing/not available/lyrics not found, в том числе повторных `NA`. `instrumental_marker` — только слово Instrumental, возможно повторённое или в скобках. `no_tokens` — нет слов длиной от двух символов. `available` означает наличие текста для векторизации, но не гарантирует полноту или достоверность lyrics в источнике. Другие короткие тексты и все языки сохранены без отбрасывания.

Для статусов, отличных от `available`, `has_lyrics=0`, а строка TF-IDF нулевая. Таких строк 296. Для текстового поиска исключайте их или учитывайте, что нулевое сходство не означает сходство текстов. Тексты не дополнялись и не генерировались.

## TF-IDF

Только lyrics. Unicode-слова по шаблону `(?u)\b\w\w+\b`, нижний регистр, униграммы. Без стоп-слов, стемминга, ограничения словаря и удаления редких слов. `tf` — число вхождений, `idf = ln((1 + N) / (1 + df)) + 1`, затем L2-нормализация строки. `N=18,158` — число доступных текстов, только они участвуют в обучении IDF. Матрица сохраняет все 18,454 строк. Словарь упорядочен лексикографически. Для нового запроса используйте тот же шаблон, сохранённый словарь и IDF, затем нормируйте вектор L2.

## Загрузка и проверка

```python
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

tracks = pd.read_csv("tracks.csv", keep_default_na=False, dtype={"track_id": str})
audio = pd.read_csv("audio_features.csv", dtype={"track_id": str})
X = load_npz("text_tfidf.npz")
assert tracks.track_id.is_unique and audio.track_id.is_unique
assert len(tracks) == len(audio) == X.shape[0]
assert tracks.track_id.tolist() == audio.track_id.tolist()
assert np.array_equal(tracks.tfidf_row.to_numpy(), np.arange(X.shape[0]))
with np.load("text_tfidf.npz", allow_pickle=False) as z:
    assert np.array_equal(tracks.track_id.to_numpy(), z["track_ids"])
    terms = z["feature_names"]  # terms[j] соответствует X[:, j]
    idf = z["idf"]
```

