function loadImageData(img, shape) {
  const pixels = tf.fromPixels(img);
  const data = tf.image.resizeBilinear(pixels, shape);
  return data.expandDims().cast('float32');
}

function displayImage(data, elemId) {
  const elem = document.getElementById(elemId);
  const pixels = tf.squeeze(data).div(tf.scalar(255));
  tf.toPixels(pixels, elem);
}

function displayLabel(probs, labels, elemId) {
  const elem = document.getElementById(elemId);
  const f = n => n.toLocaleString(undefined, { minimumFractionDigits: 2 });

  const row = labels.map((d,i) => {
    return labels[i] + ": " + f(probs[i] * 100) + "%";
  })

  elem.innerHTML = row.join("<br>");
}

// TODO port back to tfjs
function rgbToGrayscale(input) {
  // @src https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/image_ops_impl.py#L1255
  const rgbWeights = tf.tensor1d([0.2989, 0.5870, 0.1140]);
  const floatInput = input.cast('float32');
  const grayImage = floatInput.mul(rgbWeights).sum(-1);
  return grayImage.expandDims(input.shape.length - 1);
}

async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);

  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  return [topkValues, topkIndices];
}
