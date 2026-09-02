from __future__ import annotations

import gradio as gr


class WindowSelector(gr.HTML):
  def __init__(self, value: str | None = None, **kwargs) -> None:
    html_template = """
    <div class="window-selector-shell">
      <div class="window-selector-toolbar">
        <span class="selector-title">炮内软射线七号总览</span>
        <span class="selector-help">在波形上按住鼠标拖拽；蓝色为160 ms输入，红色为20 ms预测</span>
      </div>
      <canvas class="window-selector-canvas" aria-label="框选160毫秒输入时间段"></canvas>
      <div class="window-selector-status">正在载入炮内波形…</div>
    </div>
    """
    css_template = """
    .window-selector-shell {
      width: 100%;
      padding: 14px 16px 10px;
      border: 1px solid var(--border-color-primary);
      border-radius: var(--radius-lg);
      background: #ffffff;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    .window-selector-toolbar {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 8px;
      color: #111827;
    }
    .selector-title { color: #111827; font-weight: 700; font-size: 16px; }
    .selector-help { color: #64748b; font-size: 13px; text-align: right; }
    .window-selector-canvas {
      display: block;
      width: 100%;
      height: 330px;
      border-radius: 8px;
      background: #ffffff;
      cursor: crosshair;
      touch-action: none;
    }
    .window-selector-status {
      margin-top: 8px;
      color: #334155;
      font-size: 14px;
      font-weight: 600;
    }
    @media (max-width: 720px) {
      .window-selector-toolbar { display: block; }
      .selector-help { display: block; margin-top: 4px; text-align: left; }
      .window-selector-canvas { height: 285px; }
    }
    """
    js_on_load = """
    let canvas = null;
    let ctx = null;
    let payload = {};
    let dragging = false;
    let dragStartPx = 0;
    let dragCurrentPx = 0;
    const pad = {left: 62, right: 24, top: 20, bottom: 42};

    function readPayload() {
      try {
        payload = typeof props.value === 'string' ? JSON.parse(props.value) : (props.value || {});
      } catch (error) {
        payload = {};
      }
    }

    function dimensions() {
      return {
        width: Math.max(1, canvas.clientWidth),
        height: Math.max(1, canvas.clientHeight)
      };
    }

    function plotWidth() {
      return dimensions().width - pad.left - pad.right;
    }

    function timeToPixel(time) {
      const range = Math.max(1, payload.x_max - payload.x_min);
      return pad.left + (time - payload.x_min) / range * plotWidth();
    }

    function pixelToTime(pixel) {
      const range = Math.max(1, payload.x_max - payload.x_min);
      const ratio = Math.max(0, Math.min(1, (pixel - pad.left) / plotWidth()));
      return payload.x_min + ratio * range;
    }

    function resizeCanvas() {
      if (!canvas) return;
      const size = dimensions();
      const ratio = window.devicePixelRatio || 1;
      const targetWidth = Math.round(size.width * ratio);
      const targetHeight = Math.round(size.height * ratio);
      if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
      }
      ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }

    function closestAnchor(desired) {
      const anchors = payload.valid_anchors || [];
      if (!anchors.length) return null;
      let best = anchors[0];
      let distance = Math.abs(best - desired);
      for (let index = 1; index < anchors.length; index += 1) {
        const nextDistance = Math.abs(anchors[index] - desired);
        if (nextDistance < distance) {
          best = anchors[index];
          distance = nextDistance;
        }
      }
      return best;
    }

    function drawGrid(width, height, yMin, yMax) {
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.fillStyle = '#64748b';
      ctx.font = '12px sans-serif';
      for (let index = 0; index <= 6; index += 1) {
        const x = pad.left + index / 6 * (width - pad.left - pad.right);
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, height - pad.bottom);
        ctx.stroke();
        const value = payload.x_min + index / 6 * (payload.x_max - payload.x_min);
        ctx.textAlign = 'center';
        ctx.fillText(Math.round(value) + ' ms', x, height - 15);
      }
      for (let index = 0; index <= 4; index += 1) {
        const y = pad.top + index / 4 * (height - pad.top - pad.bottom);
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(width - pad.right, y);
        ctx.stroke();
        const value = yMax - index / 4 * (yMax - yMin);
        ctx.textAlign = 'right';
        ctx.fillText(value.toFixed(3), pad.left - 8, y + 4);
      }
    }

    function drawSelection(anchor, fillOnly) {
      if (anchor == null) return;
      const contextStart = anchor - payload.context_ms;
      const forecastEnd = anchor + payload.horizon_ms;
      const contextLeft = timeToPixel(contextStart);
      const anchorX = timeToPixel(anchor);
      const forecastRight = timeToPixel(forecastEnd);
      const top = pad.top;
      const height = dimensions().height - pad.top - pad.bottom;
      ctx.fillStyle = fillOnly ? 'rgba(37, 99, 235, 0.12)' : 'rgba(37, 99, 235, 0.18)';
      ctx.fillRect(contextLeft, top, anchorX - contextLeft, height);
      ctx.fillStyle = fillOnly ? 'rgba(220, 38, 38, 0.08)' : 'rgba(220, 38, 38, 0.15)';
      ctx.fillRect(anchorX, top, forecastRight - anchorX, height);
      if (!fillOnly) {
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2;
        ctx.strokeRect(contextLeft, top, anchorX - contextLeft, height);
        ctx.strokeStyle = '#dc2626';
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(anchorX, top);
        ctx.lineTo(anchorX, top + height);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    function draw() {
      if (!ctx || !payload.x || !payload.y || !payload.x.length) return;
      const size = dimensions();
      ctx.clearRect(0, 0, size.width, size.height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, size.width, size.height);
      const finiteY = payload.y.filter(value => Number.isFinite(value));
      if (!finiteY.length) return;
      let yMin = Math.min(...finiteY);
      let yMax = Math.max(...finiteY);
      const margin = Math.max((yMax - yMin) * 0.08, 1e-6);
      yMin -= margin;
      yMax += margin;
      const yToPixel = value => pad.top + (yMax - value) / (yMax - yMin)
        * (size.height - pad.top - pad.bottom);

      drawSelection(payload.selected_anchor, false);
      if (dragging) {
        const startTime = pixelToTime(Math.min(dragStartPx, dragCurrentPx));
        drawSelection(startTime + payload.context_ms, true);
      }
      drawGrid(size.width, size.height, yMin, yMax);

      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      let previousTime = null;
      for (let index = 0; index < payload.x.length; index += 1) {
        const time = payload.x[index];
        const value = payload.y[index];
        if (!Number.isFinite(value)) {
          previousTime = null;
          continue;
        }
        const x = timeToPixel(time);
        const y = yToPixel(value);
        if (previousTime === null || time - previousTime > 2) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        previousTime = time;
      }
      ctx.stroke();

      const status = element.querySelector('.window-selector-status');
      if (payload.selected_anchor != null) {
        const start = payload.selected_anchor - payload.context_ms;
        const end = payload.selected_anchor;
        const forecastEnd = end + payload.horizon_ms;
        status.textContent = '已选：' + start + '–' + end + ' ms  →  预测 ' + end
          + '–' + forecastEnd + ' ms（拖拽可重新框选）';
      } else {
        status.textContent = '请在波形上拖拽框选160 ms输入区间';
      }
    }

    function eventPixel(event) {
      const rect = canvas.getBoundingClientRect();
      return Math.max(pad.left, Math.min(rect.width - pad.right, event.clientX - rect.left));
    }

    function onPointerDown(event) {
      dragging = true;
      dragStartPx = eventPixel(event);
      dragCurrentPx = dragStartPx;
      canvas.setPointerCapture(event.pointerId);
      draw();
    }

    function onPointerMove(event) {
      if (!dragging) return;
      dragCurrentPx = eventPixel(event);
      draw();
    }

    function onPointerUp(event) {
      if (!dragging) return;
      dragCurrentPx = eventPixel(event);
      dragging = false;
      const distance = Math.abs(dragCurrentPx - dragStartPx);
      let desiredAnchor;
      if (distance < 6) desiredAnchor = pixelToTime(dragCurrentPx);
      else desiredAnchor = pixelToTime(Math.min(dragStartPx, dragCurrentPx)) + payload.context_ms;
      const anchor = closestAnchor(desiredAnchor);
      if (anchor != null) {
        payload.selected_anchor = anchor;
        draw();
        trigger('select', {shot: payload.shot, anchor_ms: anchor});
      }
    }

    function bind() {
      canvas = element.querySelector('.window-selector-canvas');
      if (!canvas || canvas.dataset.bound === 'true') {
        readPayload();
        resizeCanvas();
        return;
      }
      canvas.dataset.bound = 'true';
      canvas.addEventListener('pointerdown', onPointerDown);
      canvas.addEventListener('pointermove', onPointerMove);
      canvas.addEventListener('pointerup', onPointerUp);
      canvas.addEventListener('pointercancel', () => {
        dragging = false;
        draw();
      });
      const observer = new ResizeObserver(resizeCanvas);
      observer.observe(canvas);
      readPayload();
      resizeCanvas();
    }

    bind();
    watch('value', () => {
      readPayload();
      bind();
      draw();
    });
    """
    super().__init__(
      value=value,
      html_template=html_template,
      css_template=css_template,
      js_on_load=js_on_load,
      min_height=390,
      container=False,
      **kwargs,
    )

  def api_info(self):
    return {
      "title": "Window selector payload",
      "type": "string",
    }
