function makeCanvasDrawable(canvas) {
  const ctx = canvas.getContext('2d');
  const fillColor = "rgba(0,0,0,255)";
  ctx.fillStyle = fillColor;
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.imageSmoothingEnabled = true;

  let isDrawing = false;
  let prevPos = null;

  fillCanvas("rgba(255,255,255,255)");

  function draw(event) {
    if (isDrawing) {
      drawLineTo(getCursorPos(event));
    }
  }

  function getCursorPos(event) {
    return [event.offsetX, event.offsetY];
  }

  function drawLineTo(pos) {
    if (prevPos) {
      ctx.beginPath();
      ctx.moveTo(prevPos[0], prevPos[1]);
      ctx.lineTo(pos[0], pos[1]);
      ctx.stroke();
    }
    prevPos = pos;
  }

  const size = 10;
  function drawPixel(pos) {
    ctx.fillRect(pos[0], pos[1], size, size);
  }

  function fillCanvas(color) {
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = fillColor;
  }

  canvas.onmousedown = () => isDrawing = true;
  canvas.onmouseup = () => {
    isDrawing = false;
    prevPos = null;
  }

  return {
    start: () => canvas.onmousemove = draw,
    stop: () => canvas.onmousemove = null,
    clean: () => fillCanvas("rgba(255,255,255,255)")
  }
}
