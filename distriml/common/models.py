"""
Todos cumplen la interfaz que `algorithms.py` ya usa con los modelos de scikit-learn:

    partial_fit(X, y, classes=None) -> self      # un batch; `classes` solo en clasificación
    predict(X) -> np.ndarray

Hay dos familias:
  * Acumuladores (KNNClassifier, KNNRegressor, NearestCentroid, LeastSquares): cada `partial_fit` solo AÑADE
    información al estado (puntos guardados, sumas o contadores). El resultado no depende de cómo se parta el
    dataset en batches (salvo redondeo de coma flotante) y NO se repiten por épocas (`epochs` fijo a 1 en el
    catálogo): volver a ver los mismos datos duplicaría puntos o estadísticas.
  * Descenso de gradiente (SoftmaxRegression, LinearRegressionGD): cada `partial_fit` da UN paso sobre el batch.
    Sí se repiten por épocas, y el resultado depende del orden y del tamaño de los batches (como cualquier SGD).

En ambas, el estado completo viaja en el pickle: reanudar desde un checkpoint da exactamente el mismo modelo
que no haberse interrumpido.

X llega ya estandarizada (ver dataset.py / algorithms.py).
"""
from __future__ import annotations
import numpy as np


_BLOCK_ELEMENTS = 4_000_000     # tamaño máximo (elementos) de la matriz de distancias de cada bloque: ~32 MB


def _sq_dists(A: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Distancias euclídeas al cuadrado entre cada fila de A y cada fila de S: matriz (len(A), len(S))."""
    d = (A * A).sum(axis=1)[:, None] - 2.0 * (A @ S.T) + (S * S).sum(axis=1)[None, :]
    return np.maximum(d, 0.0)      # el redondeo puede dar negativos minúsculos


def _class_index(classes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Posición de cada etiqueta dentro de `classes` (ordenadas). ValueError si alguna no pertenece."""
    idx = np.searchsorted(classes, y)
    if np.any(idx >= len(classes)) or not np.array_equal(classes[np.minimum(idx, len(classes) - 1)], y):
        raise ValueError("el batch contiene etiquetas que no están en `classes`")
    
    return idx



class _Memory:
    """Puntos de entrenamiento acumulados, compartidos por los dos kNN."""

    def __init__(self, k: int):
        if int(k) < 1:
            raise ValueError("k debe ser >= 1")
        self.k = int(k)
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def _remember(self, X, y) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X e y deben tener el mismo número de filas")
        
        if self.X_ is None:
            self.X_, self.y_ = X.copy(), y.copy()
        else:
            self.X_ = np.vstack([self.X_, X])
            self.y_ = np.concatenate([self.y_, y])

    def _neighbours(self, X) -> np.ndarray:
        """Índices (m, k') de los k' = min(k, n_guardados) vecinos más cercanos de cada fila de X.

        Coste O(m·n) en vez de ordenar cada fila (O(m·n·log n)), y por bloques para no construir una matriz
        de distancias gigante. Ante empate de distancia en el límite gana el punto guardado antes (determinista).
        Los índices de cada fila salen en orden de almacenamiento, no de cercanía (el voto y la media no lo necesitan)."""
        if self.X_ is None:
            raise ValueError("el modelo no ha visto ningún batch")
        X = np.asarray(X, dtype=np.float64)
        n = len(self.X_)
        k = min(self.k, n)
        out = np.empty((len(X), k), dtype=np.int64)
        step = max(1, _BLOCK_ELEMENTS // n)
        for lo in range(0, len(X), step):
            d = _sq_dists(X[lo:lo + step], self.X_)
            if k == n:
                out[lo:lo + step] = np.arange(n)
                continue

            kth = np.partition(d, k - 1, axis=1)[:, k - 1:k]          # distancia del k-ésimo vecino
            closer = d < kth                                          # estrictamente más cerca: siempre entran (< k)
            tied = d == kth
            need = k - closer.sum(axis=1)                             # plazas que quedan para los empatados
            chosen = closer | (tied & (np.cumsum(tied, axis=1) <= need[:, None]))
            out[lo:lo + step] = np.nonzero(chosen)[1].reshape(-1, k)
        return out


class KNNClassifier(_Memory):
    """k vecinos más cercanos (clasificación). «Entrenar» = guardar los puntos de cada batch.

    Voto por mayoría; ante empate gana la clase de menor etiqueta. Si aún hay menos de k puntos se usan todos.
    Ojo: el estado (y por tanto cada checkpoint) crece con el número de filas vistas."""

    def __init__(self, k: int = 5):
        super().__init__(k)
        self.classes_: np.ndarray | None = None

    def partial_fit(self, X, y, classes=None):
        if self.classes_ is None:
            if classes is None:
                raise ValueError("la primera llamada a partial_fit necesita `classes`")
            self.classes_ = np.unique(np.asarray(classes))

        _class_index(self.classes_, np.asarray(y))               # valida las etiquetas
        self._remember(X, y)
        return self


    def predict(self, X) -> np.ndarray:
        nn = self._neighbours(X)
        votes = np.zeros((nn.shape[0], len(self.classes_)), dtype=np.int64)
        np.add.at(votes, (np.arange(nn.shape[0])[:, None], _class_index(self.classes_, self.y_[nn])), 1)
        return self.classes_[np.argmax(votes, axis=1)]


class KNNRegressor(_Memory):
    """k vecinos más cercanos (regresión): media del valor de los k vecinos."""

    def partial_fit(self, X, y):
        self._remember(X, np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X) -> np.ndarray:
        return self.y_[self._neighbours(X)].mean(axis=1)


class NearestCentroid:
    """Clasifica con la clase cuyo centroide (media de sus puntos) está más cerca.

    Estado mínimo: suma de características y número de puntos por clase. Una clase que aún no se ha visto
    no participa en la predicción."""

    def __init__(self):
        self.classes_: np.ndarray | None = None
        self.sum_: np.ndarray | None = None
        self.count_: np.ndarray | None = None

    def partial_fit(self, X, y, classes=None):
        X = np.asarray(X, dtype=np.float64)
        if self.classes_ is None:
            if classes is None:
                raise ValueError("la primera llamada a partial_fit necesita `classes`")
            self.classes_ = np.unique(np.asarray(classes))
            self.sum_ = np.zeros((len(self.classes_), X.shape[1]))
            self.count_ = np.zeros(len(self.classes_), dtype=np.int64)

        idx = _class_index(self.classes_, np.asarray(y))
        np.add.at(self.sum_, idx, X)
        np.add.at(self.count_, idx, 1)
        return self


    def predict(self, X) -> np.ndarray:
        if self.classes_ is None or not self.count_.any():
            raise ValueError("el modelo no ha visto ningún batch")
        seen = self.count_ > 0
        centroids = self.sum_[seen] / self.count_[seen][:, None]
        nearest = np.argmin(_sq_dists(np.asarray(X, dtype=np.float64), centroids), axis=1)   # empate: primera clase
        return self.classes_[seen][nearest]


class LeastSquares:
    """Regresión lineal por mínimos cuadrados con regularización ridge opcional (`alpha`).

    En cada batch acumula A += XbᵀXb y b += Xbᵀy (Xb = X con una columna de unos para el sesgo).
    Los pesos se resuelven al predecir: (A + alpha·I) w = b, sin penalizar el sesgo. Se usa `lstsq`
    (solución de norma mínima), que sigue siendo válida cuando aún hay menos filas que columnas."""

    def __init__(self, alpha: float = 0.0):
        if float(alpha) < 0:
            raise ValueError("alpha debe ser >= 0")
        self.alpha = float(alpha)
        self.A_: np.ndarray | None = None
        self.b_: np.ndarray | None = None
        self.n_ = 0

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        Xb = np.hstack([X, np.ones((len(X), 1))])
        if self.A_ is None:
            self.A_ = np.zeros((Xb.shape[1], Xb.shape[1]))
            self.b_ = np.zeros(Xb.shape[1])
        self.A_ += Xb.T @ Xb
        self.b_ += Xb.T @ y
        self.n_ += len(y)
        return self

    def _weights(self) -> np.ndarray:
        if self.A_ is None:
            raise ValueError("el modelo no ha visto ningún batch")
        reg = self.alpha * np.eye(len(self.b_))
        reg[-1, -1] = 0.0                                   # el sesgo no se penaliza
        return np.linalg.lstsq(self.A_ + reg, self.b_, rcond=None)[0]

    @property
    def coef_(self) -> np.ndarray:
        return self._weights()[:-1]

    @property
    def intercept_(self) -> float:
        return float(self._weights()[-1])

    def predict(self, X) -> np.ndarray:
        w = self._weights()
        return np.asarray(X, dtype=np.float64) @ w[:-1] + w[-1]


class _GradientDescent:
    """Base de los modelos entrenados por descenso de gradiente: pesos `W_`, sesgo `b_` y un paso por batch.

    La función de pérdida es la media del batch más 0.5·l2·||W||² (el sesgo no se penaliza)."""

    def __init__(self, lr: float, l2: float):
        if not float(lr) > 0:
            raise ValueError("lr debe ser > 0")
        if float(l2) < 0:
            raise ValueError("l2 debe ser >= 0")
        self.lr, self.l2 = float(lr), float(l2)
        self.W_: np.ndarray | None = None
        self.b_: np.ndarray | None = None
        self.steps_ = 0

    def _descend(self, grad_W: np.ndarray, grad_b: np.ndarray) -> None:
        self.W_ = self.W_ - self.lr * grad_W
        self.b_ = self.b_ - self.lr * grad_b
        self.steps_ += 1
        if not (np.isfinite(self.W_).all() and np.isfinite(self.b_).all()):
            # Mejor fallar que guardar un modelo de NaN: el worker lo reporta como fallo del entrenamiento.
            raise ValueError("el entrenamiento divergió (pesos no finitos): reduce `lr`")


class SoftmaxRegression(_GradientDescent):
    """Regresión logística multiclase (softmax) con descenso de gradiente por mini-batches.

    Pérdida: entropía cruzada media + 0.5·l2·||W||². Pesos iniciales a cero (el problema es convexo, así que
    no hace falta aleatoriedad y el entrenamiento es determinista). Predice la clase de mayor logit."""

    def __init__(self, lr: float = 0.5, l2: float = 0.0):
        super().__init__(lr, l2)
        self.classes_: np.ndarray | None = None

    def _logits(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None:
            raise ValueError("el modelo no ha visto ningún batch")
        return np.asarray(X, dtype=np.float64) @ self.W_ + self.b_

    def _probabilities(self, X: np.ndarray) -> np.ndarray:
        z = self._logits(X)
        e = np.exp(z - z.max(axis=1, keepdims=True))           # restar el máximo evita desbordar exp
        return e / e.sum(axis=1, keepdims=True)

    def _gradient(self, X: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gradiente de la pérdida respecto a (W, b) para un batch con etiquetas ya convertidas a índices."""
        m = len(X)
        err = self._probabilities(X)
        err[np.arange(m), idx] -= 1.0                           # P - Y (Y = one-hot)
        err /= m
        return X.T @ err + self.l2 * self.W_, err.sum(axis=0)

    def loss(self, X, y) -> float:
        z = self._logits(X)
        idx = _class_index(self.classes_, np.asarray(y))
        top = z.max(axis=1, keepdims=True)
        log_sum = np.log(np.exp(z - top).sum(axis=1)) + top[:, 0]
        return float(np.mean(log_sum - z[np.arange(len(z)), idx]) + 0.5 * self.l2 * np.sum(self.W_ ** 2))

    def partial_fit(self, X, y, classes=None):
        X = np.asarray(X, dtype=np.float64)
        if self.classes_ is None:
            if classes is None:
                raise ValueError("la primera llamada a partial_fit necesita `classes`")
            self.classes_ = np.unique(np.asarray(classes))
            self.W_ = np.zeros((X.shape[1], len(self.classes_)))
            self.b_ = np.zeros(len(self.classes_))
        idx = _class_index(self.classes_, np.asarray(y))
        with np.errstate(over="ignore", invalid="ignore"):      # si diverge, `_descend` lo detecta y avisa con un error claro
            self._descend(*self._gradient(X, idx))
        return self

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self._logits(X), axis=1)]


class LinearRegressionGD(_GradientDescent):
    """Regresión lineal con descenso de gradiente por mini-batches.

    Pérdida: 0.5·media((ŷ - y)²) + 0.5·l2·||W||². Es el mismo modelo que `LeastSquares`, pero entrenado de forma
    iterativa (con épocas) en vez de resolver las ecuaciones normales.

    Ojo con `lr`: con batch completo el descenso converge solo si lr < 2/λmax, donde λmax es el mayor autovalor de
    XᵀX/n. Con características estandarizadas λmax <= nº de características, así que lr < 2/d siempre es seguro;
    por encima puede divergir (si los pesos dejan de ser finitos se lanza ValueError y el job falla)."""

    def __init__(self, lr: float = 0.1, l2: float = 0.0):
        super().__init__(lr, l2)

    def _predict(self, X) -> np.ndarray:
        if self.W_ is None:
            raise ValueError("el modelo no ha visto ningún batch")
        return np.asarray(X, dtype=np.float64) @ self.W_ + self.b_[0]

    def _gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        resid = (self._predict(X) - y) / len(X)
        return X.T @ resid + self.l2 * self.W_, np.array([resid.sum()])

    def loss(self, X, y) -> float:
        resid = self._predict(X) - np.asarray(y, dtype=np.float64)
        return float(0.5 * np.mean(resid ** 2) + 0.5 * self.l2 * np.sum(self.W_ ** 2))

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        if self.W_ is None:
            self.W_ = np.zeros(X.shape[1])
            self.b_ = np.zeros(1)
        y = np.asarray(y, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):      # si diverge, `_descend` lo detecta y avisa con un error claro
            self._descend(*self._gradient(X, y))
        return self

    @property
    def coef_(self) -> np.ndarray:
        return self.W_

    @property
    def intercept_(self) -> float:
        return float(self.b_[0])

    def predict(self, X) -> np.ndarray:
        return self._predict(X)