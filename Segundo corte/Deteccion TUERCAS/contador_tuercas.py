"""
Contador de tuercas para banda transportadora (OpenCV)

Funciones:
  - Detecta y cuenta tuercas dentro de una zona de interes (ROI).
  - Calcula la posicion (centro X,Y) de cada tuerca.
  - Indica cuantas exceden el limite establecido.
  - Guarda configuracion en config_tuercas.json.

Requisitos:
  pip install opencv-python numpy

Uso:
  python contador_tuercas.py

Teclas:
  q  -> salir
  s  -> guardar configuracion
  r  -> seleccionar nueva zona ROI con el raton
  h  -> cambiar entre deteccion por contornos y HoughCircles
  m  -> mostrar/ocultar ventana de mascara binarizada
"""

import cv2
import numpy as np
import json
import os
import time

CONFIG_FILE = "config_tuercas.json"

DEFAULT_CONFIG = {
    "camara_id": 0,
    "limite_tuercas": 20,
    "umbral": 120,
    "area_min": 800,
    "area_max": 15000,
    "invertir": 1,
    "circularidad_min": 0.45,
    "aspecto_min": 0.65,
    "aspecto_max": 1.5,
    "roi": [0, 0, 640, 480],          # [x, y, ancho, alto]
    "usar_hough": False,
    "dp": 1.2,
    "min_dist": 40,
    "param1": 100,
    "param2": 30,
    "radio_min": 12,
    "radio_max": 70,
    "kernel_size": 5,
    "iter_close": 2,
    "iter_open": 1,
    "mostrar_mascara": True
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        except Exception:
            print("[WARN] No se pudo leer config, usando valores por defecto.")
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
        print(f"[CONFIG] Guardado en {CONFIG_FILE}")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar config: {e}")


class NutsCounter:
    def __init__(self, config):
        self.cfg = config
        self.cap = None

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.cfg["camara_id"])
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la camara {self.cfg['camara_id']}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self.cap

    def get_roi(self):
        x, y, w, h = self.cfg["roi"]
        return int(x), int(y), int(w), int(h)

    def preprocess(self, gray):
        """Escala de grises, desenfoque, umbral y limpieza morfologica."""
        cfg = self.cfg
        k = cfg["kernel_size"]
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(gray, (k, k), 0)

        if cfg["invertir"] == 1:
            _, th = cv2.threshold(blur, cfg["umbral"], 255, cv2.THRESH_BINARY_INV)
        else:
            _, th = cv2.threshold(blur, cfg["umbral"], 255, cv2.THRESH_BINARY)

        kernel = np.ones((k, k), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=cfg["iter_close"])
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=cfg["iter_open"])
        return th

    def detect_contours(self, mask, offset):
        """Deteccion basada en contornos con filtros de tamano, circularidad y aspecto."""
        cfg = self.cfg
        x_off, y_off = offset
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tuercas = []
        for c in contornos:
            area = cv2.contourArea(c)
            if not (cfg["area_min"] < area < cfg["area_max"]):
                continue

            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue

            circularidad = 4.0 * np.pi * area / (peri * peri)
            if circularidad < cfg["circularidad_min"]:
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspecto = float(w) / h
            if not (cfg["aspecto_min"] <= aspecto <= cfg["aspecto_max"]):
                continue

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            tuercas.append({
                "bbox": (x + x_off, y + y_off, w, h),
                "centro": (cx + x_off, cy + y_off),
                "centro_roi": (cx, cy),
                "area": int(area),
                "circularidad": round(float(circularidad), 3),
                "aspecto": round(aspecto, 2)
            })

        # Ordenar de arriba a abajo y de izquierda a derecha
        tuercas.sort(key=lambda t: (t["centro"][1], t["centro"][0]))
        return tuercas

    def detect_hough(self, gray, offset):
        """Deteccion alternativa basada en circulos (HoughCircles)."""
        cfg = self.cfg
        x_off, y_off = offset
        k = cfg["kernel_size"]
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(gray, (k, k), 0)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=cfg["dp"],
            minDist=cfg["min_dist"],
            param1=cfg["param1"],
            param2=cfg["param2"],
            minRadius=cfg["radio_min"],
            maxRadius=cfg["radio_max"]
        )

        tuercas = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                cx_r, cy_r, r = int(c[0]), int(c[1]), int(c[2])
                tuercas.append({
                    "bbox": (cx_r - r + x_off, cy_r - r + y_off, 2 * r, 2 * r),
                    "centro": (cx_r + x_off, cy_r + y_off),
                    "centro_roi": (cx_r, cy_r),
                    "area": int(np.pi * r * r),
                    "circularidad": 1.0,
                    "aspecto": 1.0,
                    "radio": r
                })

        tuercas.sort(key=lambda t: (t["centro"][1], t["centro"][0]))
        return tuercas

    def draw(self, frame, tuercas, excedente, fps=None):
        cfg = self.cfg
        x, y, w, h = self.get_roi()

        # Dibujar zona de interes
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, "ROI - Zona de inspeccion",
                    (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        limite = cfg["limite_tuercas"]
        for i, t in enumerate(tuercas):
            bx, by, bw, bh = t["bbox"]
            cx, cy = t["centro"]
            es_excedente = i >= limite
            color = (0, 0, 255) if es_excedente else (0, 255, 0)

            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"#{i+1} ({cx},{cy})",
                        (bx, max(by - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"A:{t['area']} C:{t['circularidad']}",
                        (bx, by + bh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        total = len(tuercas)
        exc = max(0, total - limite)
        cv2.putText(frame, f"Detectadas: {total}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Limite: {limite}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        col_exc = (0, 0, 255) if exc > 0 else (0, 255, 0)
        cv2.putText(frame, f"Excedente a retirar: {exc}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col_exc, 2)
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.putText(frame, "q=salir | s=guardar | r=ROI | h=Hough | m=mascara",
                    (20, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    def select_roi(self, frame):
        """Permite seleccionar la zona de la banda con el raton."""
        cv2.namedWindow("Seleccionar ROI", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("Seleccionar ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Seleccionar ROI")
        if roi[2] > 0 and roi[3] > 0:
            self.cfg["roi"] = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
            print(f"[ROI] Zona seleccionada: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
            save_config(self.cfg)
            return True
        return False

    def run(self):
        if self.cap is None:
            self.open_camera()

        cv2.namedWindow("Deteccion Tuercas", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Deteccion Tuercas", 960, 540)
        cv2.namedWindow("Controles", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Controles", 400, 380)

        def nothing(v):
            pass

        cv2.createTrackbar("Umbral", "Controles", self.cfg["umbral"], 255, nothing)
        cv2.createTrackbar("Area Min", "Controles", self.cfg["area_min"], 100000, nothing)
        cv2.createTrackbar("Area Max", "Controles", self.cfg["area_max"], 100000, nothing)
        cv2.createTrackbar("Invertir (0/1)", "Controles", self.cfg["invertir"], 1, nothing)

        prev_time = time.time()
        mostrar_mascara = self.cfg.get("mostrar_mascara", True)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] No se recibe frame de la camara.")
                break

            # Leer controles en tiempo real
            self.cfg["umbral"] = cv2.getTrackbarPos("Umbral", "Controles")
            self.cfg["area_min"] = cv2.getTrackbarPos("Area Min", "Controles")
            self.cfg["area_max"] = cv2.getTrackbarPos("Area Max", "Controles")
            self.cfg["invertir"] = cv2.getTrackbarPos("Invertir (0/1)", "Controles")

            if self.cfg["area_min"] >= self.cfg["area_max"]:
                self.cfg["area_max"] = self.cfg["area_min"] + 100

            x, y, w, h = self.get_roi()
            h_img, w_img = frame.shape[:2]

            # Proteger la ROI de salirse de la imagen
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))

            roi_frame = frame[y:y + h, x:x + w]
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

            if self.cfg.get("usar_hough", False):
                tuercas = self.detect_hough(gray, (x, y))
                mask = None
            else:
                mask = self.preprocess(gray)
                tuercas = self.detect_contours(mask, (x, y))

            excedente = max(0, len(tuercas) - self.cfg["limite_tuercas"])

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0
            prev_time = curr_time

            self.draw(frame, tuercas, excedente, fps)

            cv2.imshow("Deteccion Tuercas", frame)
            if mask is not None and mostrar_mascara:
                cv2.imshow("Mascara Binarizada (ROI)", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_config(self.cfg)
            elif key == ord('r'):
                self.select_roi(frame)
            elif key == ord('h'):
                self.cfg["usar_hough"] = not self.cfg.get("usar_hough", False)
                print(f"[MODO] HoughCircles = {self.cfg['usar_hough']}")
            elif key == ord('m'):
                mostrar_mascara = not mostrar_mascara
                if not mostrar_mascara:
                    cv2.destroyWindow("Mascara Binarizada (ROI)")

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    cfg = load_config()
    print("--- SISTEMA DE CONTEO DE TUERCAS (OPENCV) ---")
    print(f"Camara: {cfg['camara_id']} | Limite: {cfg['limite_tuercas']}")
    print("Controles: q=salir | s=guardar config | r=seleccionar ROI | h=cambiar a Hough | m=toggle mascara")
    try:
        counter = NutsCounter(cfg)
        counter.run()
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
