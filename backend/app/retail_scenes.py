"""Synthetic retail-context CCTV frame generator.

Renders supermarket / hypermarket / shopping-mall style scenes with shelves,
products and animated customer silhouettes. Produces numpy frames identical
in shape to those coming from real MJPEG cameras, so the rest of the capture
+ YOLO pipeline can consume them transparently.

Why synthetic? Public retail CCTV streams either need licenses or fail
intermittently. A locally rendered scene gives a deterministic, always-online
demo that still shows retail context (shelves, aisles, customers) and lets
YOLOv8 detect the rendered "person" silhouettes as actual people.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import cv2
import numpy as np

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

_PERSON_BODY = (28, 32, 44)        # dark, body silhouette
_PERSON_HIGHLIGHT = (60, 70, 90)   # rim highlight
_SHADOW = (0, 0, 0)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class CustomerSilhouette:
    """A person walking along an aisle. Positions in pixel space."""

    x: float
    y: float
    height: int
    speed: float
    direction: int = 1     # 1 = right, -1 = left
    shirt_color: tuple[int, int, int] = (90, 110, 140)
    has_cart: bool = False

    def step(self, scene_width: int) -> None:
        self.x += self.speed * self.direction
        if self.x > scene_width + 40:
            self.x = -40
        elif self.x < -40:
            self.x = scene_width + 40


@dataclass
class RetailScene:
    """One synthetic CCTV camera scene."""

    scene_id: str
    title: str
    location: str
    palette: dict[str, tuple[int, int, int]]
    shelf_products: list[tuple[int, int, int]]
    customers: list[CustomerSilhouette] = field(default_factory=list)
    background: np.ndarray | None = None
    renderer: Callable[["RetailScene", int], np.ndarray] | None = None
    frame_index: int = 0


# ---------- helpers ----------


def _bgr(r: int, g: int, b: int) -> tuple[int, int, int]:
    return (b, g, r)


def _grad_vertical(width: int, height: int, top: tuple[int, int, int], bot: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / max(1, height - 1)
        b = int(top[0] * (1 - t) + bot[0] * t)
        g = int(top[1] * (1 - t) + bot[1] * t)
        r = int(top[2] * (1 - t) + bot[2] * t)
        img[y, :] = (b, g, r)
    return img


def _draw_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float,
               color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(img, text, org, _FONT, scale, _SHADOW, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_person(img: np.ndarray, x: int, y: int, height: int,
                 shirt: tuple[int, int, int], has_cart: bool = False) -> None:
    """Draw a stylised but person-shaped silhouette so YOLO recognises it."""
    head_r = max(8, height // 8)
    body_w = max(16, height // 4)
    body_h = height - head_r * 2

    # shadow
    cv2.ellipse(img, (x, y + height // 2 + 4), (body_w, head_r // 2), 0, 0, 360, (0, 0, 0), -1)

    # legs
    leg_w = body_w // 3
    leg_h = body_h // 2
    leg_y = y + head_r + body_h - leg_h
    cv2.rectangle(img, (x - leg_w - 1, leg_y), (x - 1, leg_y + leg_h),
                  (40, 40, 50), -1)
    cv2.rectangle(img, (x + 1, leg_y), (x + leg_w + 1, leg_y + leg_h),
                  (40, 40, 50), -1)

    # body (torso)
    cv2.rectangle(img,
                  (x - body_w // 2, y + head_r),
                  (x + body_w // 2, leg_y),
                  shirt, -1)
    cv2.rectangle(img,
                  (x - body_w // 2, y + head_r),
                  (x + body_w // 2, leg_y),
                  _PERSON_HIGHLIGHT, 1)

    # head
    cv2.circle(img, (x, y - head_r // 2 + head_r),
               head_r, (210, 190, 175), -1)
    cv2.circle(img, (x, y - head_r // 2 + head_r),
               head_r, _PERSON_HIGHLIGHT, 1)

    # arms
    cv2.line(img,
             (x - body_w // 2, y + head_r + body_h // 4),
             (x - body_w // 2 - body_w // 3, y + head_r + body_h // 2),
             shirt, 4)
    cv2.line(img,
             (x + body_w // 2, y + head_r + body_h // 4),
             (x + body_w // 2 + body_w // 3, y + head_r + body_h // 2),
             shirt, 4)

    # cart
    if has_cart:
        cart_x = x + body_w
        cart_y = y + head_r + body_h // 3
        cart_w = body_w + 12
        cart_h = body_h // 2
        cv2.rectangle(img,
                      (cart_x, cart_y),
                      (cart_x + cart_w, cart_y + cart_h),
                      (160, 165, 170), 2)
        cv2.line(img,
                 (cart_x, cart_y),
                 (x + body_w // 2, y + head_r + body_h // 2),
                 (160, 165, 170), 2)
        # wheels
        cv2.circle(img, (cart_x + 4, cart_y + cart_h + 4), 4, (30, 30, 30), -1)
        cv2.circle(img, (cart_x + cart_w - 4, cart_y + cart_h + 4), 4, (30, 30, 30), -1)


def _draw_cctv_overlay(img: np.ndarray, scene: RetailScene, frame_index: int) -> None:
    h, w = img.shape[:2]
    # vignette
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.7), int(h * 0.7)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (61, 61), 0)
    alpha = mask.astype(np.float32) / 255.0
    img[:] = (img.astype(np.float32) * alpha[..., None] +
              overlay.astype(np.float32) * (1.0 - alpha[..., None])).astype(np.uint8)

    # scanline grain
    if frame_index % 4 == 0:
        noise = np.random.normal(0, 6, img.shape).astype(np.int16)
        img[:] = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # corners overlay
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    _draw_text(img, f"AEGIS-CCTV  •  {scene.scene_id.upper()}", (16, 32), 0.6, (0, 229, 255))
    _draw_text(img, scene.title, (16, 56), 0.55, (220, 230, 240))
    _draw_text(img, scene.location, (16, 78), 0.45, (160, 200, 220))
    _draw_text(img, timestamp, (w - 280, 32), 0.55, (255, 255, 255))
    _draw_text(img, "REC ●", (w - 90, 60), 0.6, (60, 230, 130))

    # frame border
    cv2.rectangle(img, (4, 4), (w - 4, h - 4), (0, 229, 255), 1)


# ---------- backgrounds ----------


def _draw_aisle_shelves(scene: RetailScene) -> np.ndarray:
    """A receding supermarket aisle with shelves on both sides."""
    palette = scene.palette
    bg = _grad_vertical(FRAME_WIDTH, FRAME_HEIGHT,
                        palette["ceiling"], palette["floor"])
    # back wall
    cv2.rectangle(bg, (480, 220), (800, 480), palette["back_wall"], -1)
    cv2.rectangle(bg, (480, 220), (800, 480), (40, 50, 70), 2)

    # ceiling perspective lines
    for offset in range(0, 800, 80):
        cv2.line(bg, (offset, 0), (640, 220), (40, 60, 90), 1)
        cv2.line(bg, (FRAME_WIDTH - offset, 0), (640, 220), (40, 60, 90), 1)

    # floor tile lines
    for y in range(540, FRAME_HEIGHT, 36):
        cv2.line(bg, (0, y), (FRAME_WIDTH, y), (24, 28, 36), 1)
    for offset in range(-FRAME_WIDTH, FRAME_WIDTH * 2, 80):
        cv2.line(bg, (offset, FRAME_HEIGHT), (640, 480), (24, 28, 36), 1)

    products = scene.shelf_products
    # left shelves with vanishing perspective
    for i in range(5):
        depth = i / 5.0
        top = int(180 + depth * 240)
        bottom = int(700 - depth * 30)
        left = int(40 + depth * 410)
        right = int(330 - depth * 80)
        # cabinet
        cv2.rectangle(bg, (left, top), (right, bottom), palette["shelf_body"], -1)
        cv2.rectangle(bg, (left, top), (right, bottom), palette["shelf_edge"], 2)
        # 4 shelves with products
        shelf_h = (bottom - top) // 4
        for row in range(4):
            sy1 = top + row * shelf_h + 6
            sy2 = top + (row + 1) * shelf_h - 4
            cv2.line(bg, (left, sy2), (right, sy2), palette["shelf_edge"], 1)
            # products
            n_products = 5
            pw = max(8, (right - left - 8) // n_products)
            for p in range(n_products):
                px1 = left + 4 + p * pw
                px2 = px1 + pw - 3
                color = products[(row * n_products + p + i) % len(products)]
                cv2.rectangle(bg, (px1, sy1 + 4), (px2, sy2 - 2), color, -1)
                cv2.rectangle(bg, (px1, sy1 + 4), (px2, sy2 - 2),
                              palette["shelf_edge"], 1)

    # right shelves mirrored
    for i in range(5):
        depth = i / 5.0
        top = int(180 + depth * 240)
        bottom = int(700 - depth * 30)
        right = int(FRAME_WIDTH - 40 - depth * 410)
        left = int(FRAME_WIDTH - 330 + depth * 80)
        cv2.rectangle(bg, (left, top), (right, bottom), palette["shelf_body"], -1)
        cv2.rectangle(bg, (left, top), (right, bottom), palette["shelf_edge"], 2)
        shelf_h = (bottom - top) // 4
        for row in range(4):
            sy1 = top + row * shelf_h + 6
            sy2 = top + (row + 1) * shelf_h - 4
            cv2.line(bg, (left, sy2), (right, sy2), palette["shelf_edge"], 1)
            n_products = 5
            pw = max(8, (right - left - 8) // n_products)
            for p in range(n_products):
                px1 = left + 4 + p * pw
                px2 = px1 + pw - 3
                color = products[(row * n_products + p + i + 3) % len(products)]
                cv2.rectangle(bg, (px1, sy1 + 4), (px2, sy2 - 2), color, -1)
                cv2.rectangle(bg, (px1, sy1 + 4), (px2, sy2 - 2),
                              palette["shelf_edge"], 1)

    # aisle sign
    cv2.rectangle(bg, (520, 110), (760, 170), (8, 12, 22), -1)
    cv2.rectangle(bg, (520, 110), (760, 170), (0, 229, 255), 2)
    _draw_text(bg, scene.title.upper(), (538, 148), 0.7, (0, 229, 255), 2)

    return bg


def _draw_checkout(scene: RetailScene) -> np.ndarray:
    palette = scene.palette
    bg = _grad_vertical(FRAME_WIDTH, FRAME_HEIGHT,
                        palette["ceiling"], palette["floor"])
    # back wall with cigarette/lottery panels
    cv2.rectangle(bg, (0, 0), (FRAME_WIDTH, 260), palette["back_wall"], -1)
    for i, color in enumerate(scene.shelf_products[:8]):
        x = 80 + i * 140
        cv2.rectangle(bg, (x, 60), (x + 110, 220), color, -1)
        cv2.rectangle(bg, (x, 60), (x + 110, 220), palette["shelf_edge"], 2)
    # floor markings
    for x in range(0, FRAME_WIDTH, 60):
        cv2.line(bg, (x, 480), (x, FRAME_HEIGHT), (40, 50, 70), 1)
    # cashier desks
    desk_color = palette["shelf_body"]
    desk_edge = palette["shelf_edge"]
    for i, x in enumerate((140, 540, 940)):
        cv2.rectangle(bg, (x, 380), (x + 220, 540), desk_color, -1)
        cv2.rectangle(bg, (x, 380), (x + 220, 540), desk_edge, 2)
        # screen
        cv2.rectangle(bg, (x + 30, 320), (x + 130, 380), (10, 18, 28), -1)
        cv2.rectangle(bg, (x + 30, 320), (x + 130, 380), (0, 229, 255), 1)
        # conveyor belt
        cv2.rectangle(bg, (x + 140, 420), (x + 220, 540), (20, 28, 40), -1)
        for s in range(6):
            cv2.line(bg, (x + 140 + s * 14, 420), (x + 140 + s * 14, 540),
                     (50, 60, 80), 1)
        # number sign
        cv2.rectangle(bg, (x + 60, 280), (x + 160, 320), (0, 26, 40), -1)
        cv2.rectangle(bg, (x + 60, 280), (x + 160, 320), (0, 229, 255), 1)
        _draw_text(bg, f"CASHIER {i+1}", (x + 70, 306), 0.5, (0, 229, 255), 1)

    return bg


def _draw_warehouse(scene: RetailScene) -> np.ndarray:
    palette = scene.palette
    bg = _grad_vertical(FRAME_WIDTH, FRAME_HEIGHT,
                        palette["ceiling"], palette["floor"])
    # steel beams overhead
    for x in range(80, FRAME_WIDTH, 220):
        cv2.line(bg, (x, 0), (x, 180), (90, 90, 100), 6)
        cv2.line(bg, (x - 3, 0), (x - 3, 180), (120, 120, 130), 1)

    # high steel racks with crates on multiple levels
    for shelf_idx, base_x in enumerate((40, 290, 540, 790, 1040)):
        for level in range(4):
            top = 200 + level * 110
            bottom = top + 100
            cv2.rectangle(bg, (base_x, top), (base_x + 220, bottom),
                          palette["shelf_body"], -1)
            cv2.rectangle(bg, (base_x, top), (base_x + 220, bottom),
                          palette["shelf_edge"], 2)
            # crates inside
            for c in range(3):
                cx1 = base_x + 10 + c * 70
                cx2 = cx1 + 60
                color = scene.shelf_products[(level * 3 + c + shelf_idx) % len(scene.shelf_products)]
                cv2.rectangle(bg, (cx1, top + 12), (cx2, bottom - 12), color, -1)
                cv2.rectangle(bg, (cx1, top + 12), (cx2, bottom - 12),
                              palette["shelf_edge"], 1)
                _draw_text(bg, f"SKU-{level}{c}{shelf_idx}",
                           (cx1 + 6, top + bottom - top - 50), 0.35, (230, 240, 250), 1)

    return bg


def _draw_entrance(scene: RetailScene) -> np.ndarray:
    palette = scene.palette
    bg = _grad_vertical(FRAME_WIDTH, FRAME_HEIGHT,
                        palette["ceiling"], palette["floor"])
    # large glass entrance
    cv2.rectangle(bg, (340, 120), (940, 560), palette["back_wall"], -1)
    cv2.rectangle(bg, (340, 120), (940, 560), palette["shelf_edge"], 3)
    # door split
    cv2.line(bg, (640, 120), (640, 560), palette["shelf_edge"], 3)
    # reflections
    for x in range(360, 940, 40):
        cv2.line(bg, (x, 140), (x - 18, 540), (60, 90, 130), 1)
    # logo banner
    cv2.rectangle(bg, (400, 60), (880, 110), (8, 12, 22), -1)
    cv2.rectangle(bg, (400, 60), (880, 110), (0, 229, 255), 2)
    _draw_text(bg, "AEGIS MALL  •  WELCOME", (430, 96), 0.8, (0, 229, 255), 2)
    # security desk silhouette
    cv2.rectangle(bg, (60, 480), (300, 600), palette["shelf_body"], -1)
    cv2.rectangle(bg, (60, 480), (300, 600), palette["shelf_edge"], 2)
    _draw_text(bg, "SECURITY", (110, 530), 0.6, (0, 229, 255), 1)
    # tile floor lines
    for y in range(560, FRAME_HEIGHT, 40):
        cv2.line(bg, (0, y), (FRAME_WIDTH, y), (40, 50, 70), 1)
    return bg


# ---------- scene factory ----------


def _make_aisle_palette(theme: str) -> dict[str, tuple[int, int, int]]:
    if theme == "frozen":
        return {
            "ceiling": _bgr(20, 30, 50),
            "floor": _bgr(15, 25, 40),
            "back_wall": _bgr(15, 50, 80),
            "shelf_body": _bgr(220, 240, 255),
            "shelf_edge": _bgr(120, 200, 255),
        }
    if theme == "produce":
        return {
            "ceiling": _bgr(40, 50, 60),
            "floor": _bgr(28, 35, 42),
            "back_wall": _bgr(50, 80, 60),
            "shelf_body": _bgr(150, 100, 60),
            "shelf_edge": _bgr(200, 160, 90),
        }
    if theme == "beverage":
        return {
            "ceiling": _bgr(15, 25, 45),
            "floor": _bgr(12, 18, 32),
            "back_wall": _bgr(8, 14, 30),
            "shelf_body": _bgr(40, 50, 80),
            "shelf_edge": _bgr(140, 220, 240),
        }
    if theme == "warehouse":
        return {
            "ceiling": _bgr(35, 40, 50),
            "floor": _bgr(22, 26, 32),
            "back_wall": _bgr(30, 36, 48),
            "shelf_body": _bgr(110, 110, 115),
            "shelf_edge": _bgr(200, 180, 60),
        }
    if theme == "checkout":
        return {
            "ceiling": _bgr(25, 35, 55),
            "floor": _bgr(18, 24, 38),
            "back_wall": _bgr(28, 38, 58),
            "shelf_body": _bgr(80, 90, 110),
            "shelf_edge": _bgr(0, 229, 255),
        }
    return {
        "ceiling": _bgr(20, 30, 55),
        "floor": _bgr(15, 22, 40),
        "back_wall": _bgr(18, 28, 48),
        "shelf_body": _bgr(60, 70, 90),
        "shelf_edge": _bgr(0, 229, 255),
    }


def _make_products(theme: str) -> list[tuple[int, int, int]]:
    if theme == "frozen":
        return [_bgr(200, 230, 255), _bgr(180, 200, 240), _bgr(120, 180, 220),
                _bgr(140, 220, 230), _bgr(220, 240, 255), _bgr(100, 160, 200)]
    if theme == "produce":
        return [_bgr(220, 60, 60), _bgr(240, 160, 50), _bgr(60, 180, 80),
                _bgr(250, 220, 60), _bgr(190, 90, 130), _bgr(120, 80, 50)]
    if theme == "beverage":
        return [_bgr(220, 50, 60), _bgr(40, 120, 220), _bgr(50, 180, 90),
                _bgr(250, 200, 90), _bgr(180, 80, 200), _bgr(60, 200, 200)]
    if theme == "warehouse":
        return [_bgr(170, 140, 90), _bgr(150, 120, 70), _bgr(190, 160, 110),
                _bgr(120, 100, 60), _bgr(160, 130, 80), _bgr(140, 110, 70)]
    if theme == "checkout":
        return [_bgr(220, 60, 60), _bgr(40, 120, 220), _bgr(250, 200, 90),
                _bgr(60, 180, 80), _bgr(180, 80, 200), _bgr(40, 60, 90),
                _bgr(160, 120, 80), _bgr(140, 200, 220)]
    return [_bgr(120, 140, 180), _bgr(180, 200, 220), _bgr(100, 150, 200)]


def _spawn_customers(theme: str, count: int = 3, base_y: int = 540) -> list[CustomerSilhouette]:
    shirts = [_bgr(220, 60, 60), _bgr(40, 120, 220), _bgr(60, 160, 80),
              _bgr(250, 180, 80), _bgr(180, 80, 200), _bgr(120, 200, 220),
              _bgr(255, 255, 255), _bgr(60, 80, 110)]
    customers: list[CustomerSilhouette] = []
    for i in range(count):
        customers.append(CustomerSilhouette(
            x=random.uniform(80, FRAME_WIDTH - 80),
            y=base_y + random.randint(-20, 20),
            height=random.randint(140, 180),
            speed=random.uniform(2.0, 4.5),
            direction=random.choice((-1, 1)),
            shirt_color=shirts[i % len(shirts)],
            has_cart=(theme in {"frozen", "produce", "beverage"}) and random.random() > 0.5,
        ))
    return customers


def _render_aisle_with_people(scene: RetailScene, frame_index: int) -> np.ndarray:
    assert scene.background is not None
    img = scene.background.copy()
    for c in scene.customers:
        c.step(FRAME_WIDTH)
        _draw_person(img, int(c.x), int(c.y - c.height), c.height, c.shirt_color, c.has_cart)
    _draw_cctv_overlay(img, scene, frame_index)
    return img


def _render_checkout_with_people(scene: RetailScene, frame_index: int) -> np.ndarray:
    assert scene.background is not None
    img = scene.background.copy()
    for c in scene.customers:
        c.step(FRAME_WIDTH)
        _draw_person(img, int(c.x), int(c.y - c.height), c.height, c.shirt_color, c.has_cart)
    _draw_cctv_overlay(img, scene, frame_index)
    return img


def _render_warehouse_with_people(scene: RetailScene, frame_index: int) -> np.ndarray:
    assert scene.background is not None
    img = scene.background.copy()
    for c in scene.customers:
        c.step(FRAME_WIDTH)
        _draw_person(img, int(c.x), int(c.y - c.height), c.height, c.shirt_color, c.has_cart)
    _draw_cctv_overlay(img, scene, frame_index)
    return img


def _render_entrance_with_people(scene: RetailScene, frame_index: int) -> np.ndarray:
    assert scene.background is not None
    img = scene.background.copy()
    for c in scene.customers:
        c.step(FRAME_WIDTH)
        _draw_person(img, int(c.x), int(c.y - c.height), c.height, c.shirt_color, c.has_cart)
    _draw_cctv_overlay(img, scene, frame_index)
    return img


def build_scenes() -> dict[str, RetailScene]:
    scenes: dict[str, RetailScene] = {}

    aisles = [
        ("cam-hypermarket-frozen", "Aisle 7 — Frozen Foods",
         "Hypermarket • Зона глубокой заморозки", "frozen"),
        ("cam-supermarket-produce", "Section 3 — Fresh Produce",
         "Supermarket • Овощи и фрукты", "produce"),
        ("cam-supermarket-beverage", "Aisle 12 — Beverages",
         "Supermarket • Напитки", "beverage"),
    ]
    for scene_id, title, location, theme in aisles:
        palette = _make_aisle_palette(theme)
        products = _make_products(theme)
        scene = RetailScene(
            scene_id=scene_id, title=title, location=location,
            palette=palette, shelf_products=products,
            customers=_spawn_customers(theme, count=3, base_y=580),
        )
        scene.background = _draw_aisle_shelves(scene)
        scene.renderer = _render_aisle_with_people
        scenes[scene_id] = scene

    checkout_palette = _make_aisle_palette("checkout")
    checkout = RetailScene(
        scene_id="cam-supermarket-checkout",
        title="Checkout Lanes 1–3",
        location="Supermarket • Кассовая линия",
        palette=checkout_palette,
        shelf_products=_make_products("checkout"),
        customers=_spawn_customers("checkout", count=4, base_y=620),
    )
    checkout.background = _draw_checkout(checkout)
    checkout.renderer = _render_checkout_with_people
    scenes[checkout.scene_id] = checkout

    warehouse_palette = _make_aisle_palette("warehouse")
    warehouse = RetailScene(
        scene_id="cam-warehouse-stock",
        title="Backroom Rack Row B",
        location="Warehouse • Складская зона",
        palette=warehouse_palette,
        shelf_products=_make_products("warehouse"),
        customers=_spawn_customers("warehouse", count=2, base_y=640),
    )
    warehouse.background = _draw_warehouse(warehouse)
    warehouse.renderer = _render_warehouse_with_people
    scenes[warehouse.scene_id] = warehouse

    entrance_palette = _make_aisle_palette("default")
    entrance = RetailScene(
        scene_id="cam-mall-entrance",
        title="Main Entrance Atrium",
        location="Shopping Mall • Центральный вход",
        palette=entrance_palette,
        shelf_products=_make_products("checkout"),
        customers=_spawn_customers("entrance", count=4, base_y=640),
    )
    entrance.background = _draw_entrance(entrance)
    entrance.renderer = _render_entrance_with_people
    scenes[entrance.scene_id] = entrance

    return scenes


_SCENES: dict[str, RetailScene] | None = None
_SCENES_LOCK_INIT_AT: float = 0.0


def get_scenes() -> dict[str, RetailScene]:
    global _SCENES, _SCENES_LOCK_INIT_AT
    if _SCENES is None:
        random.seed(int(time.time()))
        _SCENES = build_scenes()
        _SCENES_LOCK_INIT_AT = time.time()
    return _SCENES


def render_scene_frame(scene_id: str) -> np.ndarray | None:
    scenes = get_scenes()
    scene = scenes.get(scene_id)
    if scene is None or scene.renderer is None:
        return None
    scene.frame_index += 1
    return scene.renderer(scene, scene.frame_index)


__all__ = [
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "RetailScene",
    "build_scenes",
    "get_scenes",
    "render_scene_frame",
]
